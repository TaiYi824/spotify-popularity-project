import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


st.set_page_config(page_title="Prediction Lab", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/spotify_data_processed.csv")


@st.cache_resource
def train_hit_model(df: pd.DataFrame):
    data = df.copy()

    # Define a simple hit label
    data["hit_flag"] = (data["track_popularity"] >= 70).astype(int)

    # Log-transform followers for stability
    data["log_artist_followers"] = np.log10(data["artist_followers"].clip(lower=1))

    feature_cols = [
        "artist_popularity",
        "log_artist_followers",
        "track_duration_min",
        "track_number",
        "album_total_tracks",
        "track_position_ratio",
        "is_explicit",
        "album_type",
    ]

    model_df = data[feature_cols + ["hit_flag"]].dropna().copy()

    X = model_df[feature_cols]
    y = model_df["hit_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_features = [
        "artist_popularity",
        "log_artist_followers",
        "track_duration_min",
        "track_number",
        "album_total_tracks",
        "track_position_ratio",
        "is_explicit",
    ]

    categorical_features = ["album_type"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
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
            ("classifier", LogisticRegression(max_iter=2000))
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    base_rate = y.mean()

    # Coefficients for interpretation
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.named_steps["classifier"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs
    }).sort_values("coefficient", ascending=False)

    return model, auc, acc, base_rate, coef_df


df = load_data()
model, auc, acc, base_rate, coef_df = train_hit_model(df)

st.title("Prediction Lab")
st.caption("Prototype model to estimate the probability that a track reaches high popularity (hit threshold: popularity >= 70).")

# =========================
# Model summary
# =========================
m1, m2, m3 = st.columns(3)
m1.metric("ROC-AUC", f"{auc:.3f}")
m2.metric("Accuracy", f"{acc:.3f}")
m3.metric("Base Hit Rate", f"{base_rate:.1%}")

st.info("This is a baseline prototype for exploration, not a production forecasting model.")

st.markdown("---")

# =========================
# User inputs
# =========================
st.subheader("Try a Track Profile")

left_col, right_col = st.columns(2)

with left_col:
    artist_popularity = st.slider("Artist Popularity", 0, 100, 70)
    artist_followers = st.number_input("Artist Followers", min_value=1, value=1_000_000, step=10_000)
    album_type = st.selectbox("Album Type", ["album", "single", "compilation"])
    is_explicit = st.selectbox("Explicit", ["No", "Yes"])

with right_col:
    track_duration_min = st.slider("Track Duration (min)", 0.5, 8.0, 3.2, 0.1)
    album_total_tracks = st.slider("Album Total Tracks", 1, 30, 12)
    track_number = st.slider("Track Number", 1, 30, 3)

track_position_ratio = track_number / album_total_tracks if album_total_tracks > 0 else np.nan
explicit_flag = 1 if is_explicit == "Yes" else 0
log_artist_followers = np.log10(max(artist_followers, 1))

input_df = pd.DataFrame([{
    "artist_popularity": artist_popularity,
    "log_artist_followers": log_artist_followers,
    "track_duration_min": track_duration_min,
    "track_number": track_number,
    "album_total_tracks": album_total_tracks,
    "track_position_ratio": track_position_ratio,
    "is_explicit": explicit_flag,
    "album_type": album_type
}])

hit_prob = model.predict_proba(input_df)[0, 1]
lift_vs_base = hit_prob / base_rate if base_rate > 0 else np.nan

st.markdown("### Estimated Hit Probability")
p1, p2, p3 = st.columns(3)
p1.metric("Predicted Probability", f"{hit_prob:.1%}")
p2.metric("Lift vs Base Rate", f"{lift_vs_base:.2f}x")
p3.metric("Track Position Ratio", f"{track_position_ratio:.2f}")

if hit_prob >= 0.70:
    st.success("This profile looks strong relative to the dataset.")
elif hit_prob >= 0.45:
    st.warning("This profile looks moderate. Some signals are supportive, but not dominant.")
else:
    st.error("This profile looks relatively weak versus the dataset baseline.")

st.markdown("---")

# =========================
# Model interpretation
# =========================
st.subheader("Model Interpretation")

top_positive = coef_df.head(8).copy()
top_negative = coef_df.tail(8).sort_values("coefficient", ascending=True).copy()

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Strongest Positive Signals")
    st.dataframe(top_positive, use_container_width=True, hide_index=True)

with c2:
    st.markdown("#### Strongest Negative Signals")
    st.dataframe(top_negative, use_container_width=True, hide_index=True)

st.caption("Coefficient signs come from a logistic regression baseline. They show directional associations within this model, not causal effects.")

st.markdown("---")

# =========================
# Reference notes
# =========================
st.subheader("What this prototype does well")
st.write(
    """
    - Gives an interpretable baseline hit-probability estimate
    - Lets users test different track profiles interactively
    - Shows whether artist scale, album type, explicit flag, and track placement shift hit likelihood
    """
)

st.subheader("What this prototype does not do yet")
st.write(
    """
    - It does not model release timing, marketing spend, playlist placement, or social virality
    - It does not predict exact popularity score
    - It should not be treated as a causal or production-grade recommender
    """
)