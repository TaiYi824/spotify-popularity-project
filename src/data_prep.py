import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # standardize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # parse release date
    df["album_release_date"] = pd.to_datetime(df["album_release_date"], errors="coerce")

    # basic missing handling
    df["artist_name"] = df["artist_name"].fillna("Unknown Artist")
    df["artist_genres"] = df["artist_genres"].fillna("Unknown")

    # numeric safety
    numeric_cols = [
        "track_number",
        "track_popularity",
        "artist_popularity",
        "artist_followers",
        "album_total_tracks",
        "track_duration_min",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # derived features
    df["release_year"] = df["album_release_date"].dt.year
    df["release_month"] = df["album_release_date"].dt.month
    df["is_single"] = (df["album_type"].str.lower() == "single").astype(int)
    df["is_explicit"] = df["explicit"].astype(int)

    # track position ratio inside album
    df["track_position_ratio"] = np.where(
        df["album_total_tracks"] > 0,
        df["track_number"] / df["album_total_tracks"],
        np.nan
    )

    # simple followers tier
    df["followers_tier"] = pd.qcut(
        df["artist_followers"],
        q=4,
        labels=["Low", "Mid-Low", "Mid-High", "High"],
        duplicates="drop"
    )

    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)