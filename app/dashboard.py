import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Spotify Popularity Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/spotify_data_processed.csv")
    return df


def format_large_number(x):
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


df = load_data()

# Basic guardrails
df = df.copy()
df = df.dropna(subset=["release_year", "track_popularity", "artist_popularity", "track_duration_min"])
df["release_year"] = df["release_year"].astype(int)

st.title("What Drives Spotify Track Popularity?")
st.caption("An interactive analytics dashboard built from track-, artist-, album-, and time-level features.")

# =========================
# Sidebar
# =========================
st.sidebar.header("Filters")

album_type_options = sorted(df["album_type"].dropna().unique())

album_type = st.sidebar.multiselect(
    "Album Type",
    options=album_type_options,
    default=album_type_options
)

min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())

year_range = st.sidebar.slider(
    "Release Year",
    min_year,
    max_year,
    (min_year, max_year)
)

filtered = df[
    (df["album_type"].isin(album_type)) &
    (df["release_year"].between(year_range[0], year_range[1]))
].copy()

if filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# =========================
# KPI row
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Tracks", f"{len(filtered):,}")
col2.metric("Avg Popularity", f"{filtered['track_popularity'].mean():.1f}")
col3.metric("Avg Artist Popularity", f"{filtered['artist_popularity'].mean():.1f}")
col4.metric("Avg Duration (min)", f"{filtered['track_duration_min'].mean():.2f}")

st.markdown("---")

# =========================
# Row 1 charts
# =========================
left_col, right_col = st.columns(2)

with left_col:
    fig1 = px.histogram(
        filtered,
        x="track_popularity",
        nbins=30,
        title="Distribution of Track Popularity"
    )
    fig1.update_layout(
        xaxis_title="Track Popularity",
        yaxis_title="Count"
    )
    st.plotly_chart(fig1, use_container_width=True)

with right_col:
    fig4 = px.box(
        filtered,
        x="album_type",
        y="track_popularity",
        color="album_type",
        title="Track Popularity Distribution by Album Type"
    )
    fig4.update_layout(
        xaxis_title="Album Type",
        yaxis_title="Track Popularity",
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# Row 2 charts
# =========================
left_col, right_col = st.columns(2)

with left_col:
    fig2 = px.scatter(
        filtered,
        x="artist_followers",
        y="track_popularity",
        color="album_type",
        hover_data=["track_name", "artist_name"],
        title="Artist Followers vs Track Popularity",
        log_x=True,
        opacity=0.55
    )
    fig2.update_layout(
        xaxis_title="Artist Followers (log scale)",
        yaxis_title="Track Popularity"
    )
    st.plotly_chart(fig2, use_container_width=True)

with right_col:
    year_summary = (
        filtered.groupby("release_year", as_index=False)
        .agg(
            avg_popularity=("track_popularity", "mean"),
            track_count=("track_name", "count")
        )
        .sort_values("release_year")
    )

    year_summary = year_summary[year_summary["track_count"] >= 20]

    fig3 = px.line(
        year_summary,
        x="release_year",
        y="avg_popularity",
        title="Average Track Popularity by Release Year (min 20 tracks per year)",
        markers=True
    )
    fig3.update_layout(
        xaxis_title="Release Year",
        yaxis_title="Average Track Popularity"
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# =========================
# Popularity Drivers
# =========================
st.header("Popularity Drivers")

# Create a log-scale followers feature for more stable correlation analysis
filtered["log_artist_followers"] = np.log10(filtered["artist_followers"].clip(lower=1))

def safe_corr(x, y):
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 2:
        return np.nan
    return s.iloc[:, 0].corr(s.iloc[:, 1])

corr_artist_pop = safe_corr(filtered["track_popularity"], filtered["artist_popularity"])
corr_log_followers = safe_corr(filtered["track_popularity"], filtered["log_artist_followers"])
corr_duration = safe_corr(filtered["track_popularity"], filtered["track_duration_min"])

album_type_summary = (
    filtered.groupby("album_type", as_index=False)["track_popularity"]
    .mean()
    .sort_values("track_popularity", ascending=False)
)

best_album_type = album_type_summary.iloc[0]["album_type"]
best_album_type_score = album_type_summary.iloc[0]["track_popularity"]

d1, d2, d3, d4 = st.columns(4)
d1.metric("Corr: Artist Popularity", f"{corr_artist_pop:.2f}")
d2.metric("Corr: Log Followers", f"{corr_log_followers:.2f}")
d3.metric("Corr: Duration", f"{corr_duration:.2f}")
d4.metric("Best Avg Album Type", f"{best_album_type}", f"{best_album_type_score:.1f} avg pop")

st.caption("These are descriptive relationships, not causal claims.")

# Correlation heatmap
corr_features = [
    "track_popularity",
    "artist_popularity",
    "log_artist_followers",
    "track_duration_min",
    "track_number",
    "album_total_tracks",
    "track_position_ratio"
]

corr_data = filtered[corr_features].dropna()
corr_matrix = corr_data.corr(numeric_only=True)

fig_heatmap = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="Blues",
    title="Correlation Heatmap of Popularity-Related Features"
)
fig_heatmap.update_layout(coloraxis_colorbar_title="Corr")
st.plotly_chart(fig_heatmap, use_container_width=True)

left_col, right_col = st.columns(2)

with left_col:
    if "followers_tier" in filtered.columns:
        followers_summary = (
            filtered.groupby("followers_tier", as_index=False, observed=False)["track_popularity"]
            .mean()
        )

        fig_followers_tier = px.bar(
            followers_summary,
            x="followers_tier",
            y="track_popularity",
            color="followers_tier",
            title="Average Track Popularity by Artist Followers Tier"
        )
        fig_followers_tier.update_layout(
            xaxis_title="Followers Tier",
            yaxis_title="Average Track Popularity",
            showlegend=False
        )
        st.plotly_chart(fig_followers_tier, use_container_width=True)

with right_col:
    if "is_explicit" in filtered.columns:
        explicit_summary = (
            filtered.groupby("is_explicit", as_index=False)["track_popularity"]
            .mean()
        )
        explicit_summary["explicit_label"] = explicit_summary["is_explicit"].map({
            0: "Non-Explicit",
            1: "Explicit"
        })

        fig_explicit = px.bar(
            explicit_summary,
            x="explicit_label",
            y="track_popularity",
            color="explicit_label",
            title="Average Track Popularity: Explicit vs Non-Explicit"
        )
        fig_explicit.update_layout(
            xaxis_title="Track Type",
            yaxis_title="Average Track Popularity",
            showlegend=False
        )
        st.plotly_chart(fig_explicit, use_container_width=True)

st.markdown("---")


# =========================
# Top 10 table
# =========================
st.subheader("Top 10 Tracks by Popularity")

top_tracks = (
    filtered[
        [
            "track_name",
            "artist_name",
            "track_popularity",
            "artist_popularity",
            "artist_followers",
            "album_type",
            "release_year",
            "track_duration_min"
        ]
    ]
    .sort_values(
        ["track_popularity", "artist_followers"],
        ascending=[False, False]
    )
    .head(10)
    .copy()
)

top_tracks["artist_followers"] = top_tracks["artist_followers"].apply(format_large_number)

st.dataframe(
    top_tracks.rename(
        columns={
            "track_name": "Track",
            "artist_name": "Artist",
            "track_popularity": "Track Popularity",
            "artist_popularity": "Artist Popularity",
            "artist_followers": "Artist Followers",
            "album_type": "Album Type",
            "release_year": "Release Year",
            "track_duration_min": "Duration (min)"
        }
    ),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# =========================
# Sample records
# =========================
st.subheader("Filtered Sample Records")

sample_records = (
    filtered[
        [
            "track_name",
            "artist_name",
            "track_popularity",
            "artist_popularity",
            "artist_followers",
            "album_type",
            "release_year",
            "track_duration_min"
        ]
    ]
    .sort_values(
        ["track_popularity", "artist_followers"],
        ascending=[False, False]
    )
    .head(50)
    .copy()
)

sample_records["artist_followers"] = sample_records["artist_followers"].apply(format_large_number)

st.dataframe(
    sample_records.rename(
        columns={
            "track_name": "Track",
            "artist_name": "Artist",
            "track_popularity": "Track Popularity",
            "artist_popularity": "Artist Popularity",
            "artist_followers": "Artist Followers",
            "album_type": "Album Type",
            "release_year": "Release Year",
            "track_duration_min": "Duration (min)"
        }
    ),
    use_container_width=True,
    hide_index=True
)