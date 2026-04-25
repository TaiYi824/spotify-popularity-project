from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Real-World Monitor", layout="wide")

LATEST_CSV = Path("data/live/lastfm_top_tracks_latest.csv")
SNAPSHOTS_CSV = Path("data/live/lastfm_top_tracks_snapshots.csv")
EVAL_CSV = Path("data/live/real_world_eval_scored.csv")
METRICS_JSON = Path("data/live/real_world_eval_metrics.json")


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_large_number(x: float | int | None) -> str:
    if pd.isna(x):
        return "N/A"
    x = float(x)
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B"
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================
# Page header
# =========================
st.title("Real-World Music Monitor")
st.caption(
    "This page shows how the project moves beyond a static Kaggle analysis. "
    "It collects repeated live chart snapshots from external sources, monitors what changes over time, "
    "and evaluates whether the model works in a real-world setting."
)

st.info(
    "What makes this page different from the historical dashboard?\n\n"
    "- It uses external live data instead of only the original training dataset\n"
    "- It stores repeated snapshots over time\n"
    "- It evaluates model performance on future observed outcomes\n"
    "- It creates a practical monitoring loop: collect -> score -> observe -> evaluate -> improve"
)

if not LATEST_CSV.exists():
    st.warning("Live snapshot not found yet. Run scripts/real_world_data_pipeline.py first.")
    st.stop()

latest = load_csv(LATEST_CSV)
latest = safe_numeric(
    latest,
    [
        "chart_rank",
        "lastfm_duration_ms",
        "lastfm_listeners",
        "lastfm_playcount",
        "musicbrainz_first_release_year",
    ],
)

if "lastfm_listeners" in latest.columns:
    latest["log_lastfm_listeners"] = np.log10(latest["lastfm_listeners"].clip(lower=1))

if "lastfm_playcount" in latest.columns:
    latest["log_lastfm_playcount"] = np.log10(latest["lastfm_playcount"].clip(lower=1))

if "lastfm_duration_ms" in latest.columns:
    latest["track_duration_min"] = latest["lastfm_duration_ms"] / 60000.0

if "musicbrainz_first_release_year" in latest.columns:
    latest["track_age_years_proxy"] = pd.Timestamp.utcnow().year - latest["musicbrainz_first_release_year"]

if "chart_rank" in latest.columns:
    latest["chart_rank_inv"] = 1 / latest["chart_rank"].clip(lower=1)

# =========================
# Section 1: Latest live snapshot overview
# =========================
st.header("1. Latest Live Snapshot Overview")
st.caption(
    "This section summarizes the most recent live chart data collected by the system."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tracks in Latest Snapshot", f"{len(latest):,}")
c2.metric(
    "Median Listeners",
    format_large_number(latest["lastfm_listeners"].median()) if "lastfm_listeners" in latest.columns else "N/A"
)
c3.metric(
    "Median Playcount",
    format_large_number(latest["lastfm_playcount"].median()) if "lastfm_playcount" in latest.columns else "N/A"
)
c4.metric(
    "Snapshot Date",
    str(latest["snapshot_date"].iloc[0]) if "snapshot_date" in latest.columns and len(latest) > 0 else "N/A"
)

st.markdown("---")

left_col, right_col = st.columns(2)

with left_col:
    if {"chart_rank", "lastfm_listeners", "track_name", "artist_name"}.issubset(latest.columns):
        hover_cols = [c for c in ["track_name", "artist_name", "musicbrainz_primary_type"] if c in latest.columns]
        fig_rank = px.scatter(
            latest,
            x="chart_rank",
            y="lastfm_listeners",
            hover_data=hover_cols,
            title="Chart Rank vs Listener Scale",
            log_y=True,
        )
        fig_rank.update_layout(
            xaxis_title="Chart Rank (smaller rank = stronger position)",
            yaxis_title="Last.fm Listeners (log scale)"
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        st.caption(
            "Interpretation: this chart shows whether tracks in better chart positions also tend to have larger listener bases."
        )

with right_col:
    if {"musicbrainz_primary_type", "lastfm_listeners"}.issubset(latest.columns):
        release_type_summary = (
            latest.groupby("musicbrainz_primary_type", dropna=False)["lastfm_listeners"]
            .median()
            .reset_index()
            .sort_values("lastfm_listeners", ascending=False)
        )
        fig_type = px.bar(
            release_type_summary,
            x="musicbrainz_primary_type",
            y="lastfm_listeners",
            title="Median Listener Scale by Release Type",
        )
        fig_type.update_layout(
            xaxis_title="Release Type",
            yaxis_title="Median Listeners"
        )
        st.plotly_chart(fig_type, use_container_width=True)
        st.caption(
            "Interpretation: this chart compares typical listener scale across albums, singles, EPs, and other release types."
        )

st.markdown("---")

# =========================
# Section 2: Snapshot history
# =========================
st.header("2. Snapshot History")
st.caption(
    "This section proves that the system is not based on a one-time pull. "
    "It stores repeated snapshots over time, which makes post-deployment evaluation possible."
)

if SNAPSHOTS_CSV.exists():
    snapshots = load_csv(SNAPSHOTS_CSV)
    snapshots = safe_numeric(
        snapshots,
        [
            "chart_rank",
            "lastfm_listeners",
            "lastfm_playcount",
            "musicbrainz_first_release_year",
        ],
    )

    if {"snapshot_date", "track_key"}.issubset(snapshots.columns):
        snapshot_summary = (
            snapshots.groupby("snapshot_date", as_index=False)
            .agg(track_count=("track_key", "count"))
            .sort_values("snapshot_date")
        )

        if "lastfm_listeners" in snapshots.columns:
            listeners_summary = (
                snapshots.groupby("snapshot_date", as_index=False)
                .agg(median_listeners=("lastfm_listeners", "median"))
            )
            snapshot_summary = snapshot_summary.merge(listeners_summary, on="snapshot_date", how="left")

        if "lastfm_playcount" in snapshots.columns:
            playcount_summary = (
                snapshots.groupby("snapshot_date", as_index=False)
                .agg(median_playcount=("lastfm_playcount", "median"))
            )
            snapshot_summary = snapshot_summary.merge(playcount_summary, on="snapshot_date", how="left")

        h1, h2 = st.columns(2)

        with h1:
            fig_hist_count = px.line(
                snapshot_summary,
                x="snapshot_date",
                y="track_count",
                markers=True,
                title="Number of Tracks Collected per Snapshot"
            )
            fig_hist_count.update_layout(
                xaxis_title="Snapshot Date",
                yaxis_title="Track Count"
            )
            st.plotly_chart(fig_hist_count, use_container_width=True)

        with h2:
            if "median_listeners" in snapshot_summary.columns:
                fig_hist_listeners = px.line(
                    snapshot_summary,
                    x="snapshot_date",
                    y="median_listeners",
                    markers=True,
                    title="Median Listener Scale Across Snapshots"
                )
                fig_hist_listeners.update_layout(
                    xaxis_title="Snapshot Date",
                    yaxis_title="Median Listeners"
                )
                st.plotly_chart(fig_hist_listeners, use_container_width=True)

        st.success(
            "Why this matters: once we have repeated snapshots, we can compare today's prediction with what actually happens next. "
            "That is what turns the project into a real-world monitoring system."
        )
else:
    st.info("Snapshot history file not found yet.")

st.markdown("---")

# =========================
# Section 3: Model evaluation
# =========================
st.header("3. Model Evaluation on Real-World Outcomes")
st.caption(
    "The model is evaluated on whether a track remains on the next observed chart snapshot. "
    "This is more realistic than evaluating only on the original historical dataset."
)

if EVAL_CSV.exists():
    scored = load_csv(EVAL_CSV)
    scored = safe_numeric(
        scored,
        [
            "chart_rank",
            "lastfm_listeners",
            "lastfm_playcount",
            "pred_prob_stays_next_snapshot",
            "stays_on_chart_next_snapshot",
        ],
    )

    if "pred_prob_stays_next_snapshot" in scored.columns:
        scored = scored.sort_values("pred_prob_stays_next_snapshot", ascending=False)

    if METRICS_JSON.exists():
        metrics = load_metrics(METRICS_JSON)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}")
        m2.metric("Precision", f"{metrics.get('precision', float('nan')):.3f}")
        m3.metric("Recall", f"{metrics.get('recall', float('nan')):.3f}")
        m4.metric("Brier Score", f"{metrics.get('brier_score', float('nan')):.3f}")

        m5, m6, m7 = st.columns(3)
        m5.metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.3f}")
        m6.metric("Precision@Top 10%", f"{metrics.get('precision_at_top_10pct', float('nan')):.3f}")
        m7.metric("Evaluation Sample Size", f"{int(metrics.get('sample_size', 0)):,}")

        st.caption(
            "How to read these metrics:\n"
            "- ROC-AUC: can the model rank stronger vs weaker cases well?\n"
            "- Precision: when the model predicts 'likely to stay', how often is it right?\n"
            "- Recall: how many of the actual staying tracks did the model capture?\n"
            "- Brier Score: are the predicted probabilities well calibrated? Lower is better."
        )

    st.subheader("Top Tracks by Predicted Probability")
    st.caption(
        "These tracks are ranked by the model's estimated probability of staying on the next observed snapshot."
    )

    display_cols = [
        "artist_name",
        "track_name",
        "chart_rank",
        "lastfm_listeners",
        "lastfm_playcount",
        "musicbrainz_primary_type",
        "pred_prob_stays_next_snapshot",
        "stays_on_chart_next_snapshot",
    ]
    display_cols = [c for c in display_cols if c in scored.columns]

    show = scored[display_cols].copy()

    if "lastfm_listeners" in show.columns:
        show["lastfm_listeners"] = show["lastfm_listeners"].apply(format_large_number)
    if "lastfm_playcount" in show.columns:
        show["lastfm_playcount"] = show["lastfm_playcount"].apply(format_large_number)
    if "pred_prob_stays_next_snapshot" in show.columns:
        show["pred_prob_stays_next_snapshot"] = show["pred_prob_stays_next_snapshot"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

    rename_map = {
        "artist_name": "Artist",
        "track_name": "Track",
        "chart_rank": "Current Rank",
        "lastfm_listeners": "Listeners",
        "lastfm_playcount": "Playcount",
        "musicbrainz_primary_type": "Release Type",
        "pred_prob_stays_next_snapshot": "Predicted Stay Probability",
        "stays_on_chart_next_snapshot": "Actually Stayed Next Snapshot",
    }

    st.dataframe(
        show.head(20).rename(columns=rename_map),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info(
        "Evaluation file not found yet. After collecting repeated snapshots, run training and evaluation scripts."
    )

st.markdown("---")

# =========================
# Section 4: Latest snapshot detail
# =========================
st.header("4. Latest Snapshot Detail")
st.caption(
    "This table shows the latest live snapshot collected by the system."
)

latest_cols = [
    "artist_name",
    "track_name",
    "chart_rank",
    "lastfm_listeners",
    "lastfm_playcount",
    "musicbrainz_primary_type",
    "musicbrainz_release_date",
]
latest_cols = [c for c in latest_cols if c in latest.columns]

latest_display = latest[latest_cols].copy()

if "lastfm_listeners" in latest_display.columns:
    latest_display["lastfm_listeners"] = latest_display["lastfm_listeners"].apply(format_large_number)
if "lastfm_playcount" in latest_display.columns:
    latest_display["lastfm_playcount"] = latest_display["lastfm_playcount"].apply(format_large_number)

rename_latest = {
    "artist_name": "Artist",
    "track_name": "Track",
    "chart_rank": "Current Rank",
    "lastfm_listeners": "Listeners",
    "lastfm_playcount": "Playcount",
    "musicbrainz_primary_type": "Release Type",
    "musicbrainz_release_date": "Release Date",
}

st.dataframe(
    latest_display.rename(columns=rename_latest),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# =========================
# Section 5: What should be improved next
# =========================
st.header("5. Next Improvement Steps")
st.caption(
    "This section explains how the real-world system can be improved in future iterations."
)

st.write(
    """
    **Planned next steps**
    
    1. **Collect more snapshot dates**  
    More time points will make evaluation results more stable and more convincing.
    
    2. **Try richer target definitions**  
    Instead of only predicting whether a track stays on the next snapshot, future versions can test:
    - whether rank improves
    - whether a track enters a top segment
    - whether listener scale grows meaningfully
    
    3. **Calibrate probabilities**  
    As more live observations accumulate, predicted probabilities can be calibrated to become more reliable.
    
    4. **Run error analysis**  
    Future iterations should study false positives and false negatives to understand where the model fails.
    
    5. **Expand external features**  
    Additional real-world signals can be added if more external data sources become available.
    """
)

st.success(
    "Summary: this page demonstrates that the project now includes real-world data collection, repeated monitoring, "
    "future-outcome evaluation, and a clear optimization roadmap."
)