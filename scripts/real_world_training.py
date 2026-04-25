from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_INPUT = "data/live/lastfm_top_tracks_snapshots.csv"
DEFAULT_MODEL = "data/live/real_world_model.joblib"
DEFAULT_METADATA = "data/live/real_world_model_metadata.json"


def load_snapshots(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Snapshot file is empty.")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def build_training_panel(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.sort_values(["track_key", "snapshot_date"])

    # Dedupe within the same snapshot if it was collected more than once.
    work = work.drop_duplicates(subset=["snapshot_date", "track_key"], keep="last")

    # Feature engineering
    work["log_lastfm_listeners"] = np.log10(work["lastfm_listeners"].clip(lower=1))
    work["log_lastfm_playcount"] = np.log10(work["lastfm_playcount"].clip(lower=1))
    work["chart_rank_inv"] = 1 / work["chart_rank"].clip(lower=1)
    work["track_duration_min"] = work["lastfm_duration_ms"] / 60000.0

    snap_ts = work["snapshot_date"]
    first_release_year = work["musicbrainz_first_release_year"]
    work["track_age_years_proxy"] = snap_ts.dt.year - first_release_year
    work["track_age_years_proxy"] = work["track_age_years_proxy"].where(work["track_age_years_proxy"] >= 0)

    # Label: does this track remain on the next collected chart snapshot?
    work["next_snapshot_date"] = work.groupby("track_key")["snapshot_date"].shift(-1)
    next_global_dates = sorted(work["snapshot_date"].dropna().unique())
    date_map = {d: next_global_dates[i + 1] if i + 1 < len(next_global_dates) else pd.NaT for i, d in enumerate(next_global_dates)}
    work["expected_next_snapshot_date"] = work["snapshot_date"].map(date_map)
    work["stays_on_chart_next_snapshot"] = (
        work["next_snapshot_date"] == work["expected_next_snapshot_date"]
    ).astype(float)

    # Only rows with an observable next snapshot can be used for supervised training.
    work = work[work["expected_next_snapshot_date"].notna()].copy()

    return work


def train_model(panel: pd.DataFrame) -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    feature_cols = [
        "chart_rank",
        "chart_rank_inv",
        "log_lastfm_listeners",
        "log_lastfm_playcount",
        "track_duration_min",
        "track_age_years_proxy",
        "musicbrainz_primary_type",
    ]
    target_col = "stays_on_chart_next_snapshot"

    model_df = panel[feature_cols + [target_col]].copy()
    model_df = model_df.dropna(subset=[target_col])
    if model_df[target_col].nunique() < 2:
        raise ValueError(
            "Need at least two classes in the target. Collect more than one or two snapshots before training."
        )

    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)

    numeric_features = [
        "chart_rank",
        "chart_rank_inv",
        "log_lastfm_listeners",
        "log_lastfm_playcount",
        "track_duration_min",
        "track_age_years_proxy",
    ]
    categorical_features = ["musicbrainz_primary_type"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "base_rate": float(y.mean()),
        "n_rows": float(len(model_df)),
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
    }

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_df = coef_df.sort_values("coefficient", ascending=False)

    return model, metrics, coef_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a live-data model on Last.fm/MusicBrainz snapshots.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model-out", default=DEFAULT_MODEL)
    parser.add_argument("--metadata-out", default=DEFAULT_METADATA)
    parser.add_argument("--coef-out", default="data/live/real_world_model_coefficients.csv")
    args = parser.parse_args()

    snapshots = load_snapshots(args.input)
    panel = build_training_panel(snapshots)
    if len(panel) < 50:
        raise SystemExit(
            "Not enough labeled rows yet. Collect more repeated snapshots over time before training."
        )

    model, metrics, coef_df = train_model(panel)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    coef_df.to_csv(args.coef_out, index=False)

    metadata = {
        "label_definition": "1 if the track stays on the next collected chart snapshot, else 0",
        "metrics": metrics,
        "input_snapshot_file": args.input,
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print("\nTop positive coefficients:")
    print(coef_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
