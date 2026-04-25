from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score

from real_world_training import build_training_panel, load_snapshots

DEFAULT_INPUT = "data/live/lastfm_top_tracks_snapshots.csv"
DEFAULT_MODEL = "data/live/real_world_model.joblib"
DEFAULT_METRICS = "data/live/real_world_eval_metrics.json"
DEFAULT_SCORED = "data/live/real_world_eval_scored.csv"


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    order = np.argsort(-y_prob)
    top_idx = order[:k]
    return float(np.mean(y_true[top_idx])) if len(top_idx) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the live-data model.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS)
    parser.add_argument("--scored-out", default=DEFAULT_SCORED)
    args = parser.parse_args()

    snapshots = load_snapshots(args.input)
    panel = build_training_panel(snapshots)
    if panel.empty:
        raise SystemExit("No evaluable labeled rows found. Collect more snapshots first.")

    model = joblib.load(args.model)

    feature_cols = [
        "chart_rank",
        "chart_rank_inv",
        "log_lastfm_listeners",
        "log_lastfm_playcount",
        "track_duration_min",
        "track_age_years_proxy",
        "musicbrainz_primary_type",
    ]
    scored = panel.copy()
    scored["pred_prob_stays_next_snapshot"] = model.predict_proba(scored[feature_cols])[:, 1]
    scored["pred_label_stays_next_snapshot"] = (scored["pred_prob_stays_next_snapshot"] >= 0.50).astype(int)

    y_true = scored["stays_on_chart_next_snapshot"].astype(int).to_numpy()
    y_prob = scored["pred_prob_stays_next_snapshot"].to_numpy()
    y_pred = scored["pred_label_stays_next_snapshot"].to_numpy()

    n_top = max(5, int(len(scored) * 0.10))
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision_at_top_10pct": precision_at_k(y_true, y_prob, n_top),
        "sample_size": int(len(scored)),
        "positive_rate": float(np.mean(y_true)),
    }

    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scored.to_csv(args.scored_out, index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
