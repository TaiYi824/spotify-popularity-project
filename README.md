# What Drives Spotify Track Popularity?

An interactive analytics and baseline prediction project that explores how track-, artist-, album-, and time-level features relate to Spotify track popularity.

![Project Banner Placeholder](outputs/figures/project_banner.png)

## Results Snapshot

- **Tracks analyzed:** 8,582
- **Baseline hit definition:** `track_popularity >= 70`
- **Model:** Logistic Regression
- **ROC-AUC:** 0.740
- **Accuracy:** 0.718
- **Base hit rate:** 27.7%

## Why this project

This project was built as a **portfolio-style analytics product**, not just a notebook exercise.

It combines:
- an interactive **Streamlit dashboard**
- a **Popularity Drivers** analysis section
- a **Prediction Lab** that estimates hit probability from a user-defined track profile

The goal is to show how a cleaned dataset can be turned into a small but credible analytics application with both **descriptive insight** and **baseline predictive modeling**.

---

## Project Highlights

- Built an interactive dashboard for filtering and exploring Spotify track popularity
- Engineered derived features such as release year, explicit flag, track position ratio, and follower tiers
- Added a driver-analysis layer with correlations, segment comparisons, and heatmap views
- Developed a baseline **hit probability prototype** using logistic regression
- Designed the project to be reproducible and portfolio-ready with preprocessing, app structure, and documentation

---

## Main Questions

This project explores questions such as:

- How is track popularity distributed?
- Do artist popularity and follower count relate to track popularity?
- Do album type, track placement, track duration, or explicit content matter?
- Can a simple baseline model estimate whether a track is likely to become a “hit”?

---

## App Overview

### 1) Overview Dashboard
The main dashboard allows users to:

- filter tracks by album type and release year
- review KPI summaries
- inspect popularity distributions
- compare album-type performance
- examine the relationship between followers and popularity
- review top tracks and filtered records

### 2) Popularity Drivers
This section adds a more analytical layer through:

- descriptive correlation metrics
- a correlation heatmap
- average popularity by followers tier
- average popularity for explicit vs non-explicit tracks

### 3) Prediction Lab
The second page provides a baseline predictive prototype that:

- defines a “hit” as `track_popularity >= 70`
- trains a logistic regression model using artist-, track-, and album-level inputs
- estimates hit probability for a user-defined track profile
- displays model coefficients for simple interpretation

---

## Screenshots

### Overview Dashboard
![Overview Dashboard](outputs/figures/dashboard_overview.png)

### Prediction Lab
![Prediction Lab](outputs/figures/prediction_lab.png)

---

## Dataset

The project starts from a cleaned Spotify CSV file and uses fields such as:

- `track_popularity`
- `artist_popularity`
- `artist_followers`
- `album_type`
- `album_release_date`
- `track_number`
- `album_total_tracks`
- `track_duration_min`
- `explicit`

### Engineered features
Additional features created during preprocessing include:

- `release_year`
- `release_month`
- `is_single`
- `is_explicit`
- `track_position_ratio`
- `followers_tier`
- `log_artist_followers`

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Plotly Express**
- **Streamlit**
- **scikit-learn**

---

## Project Structure

```text
spotify-popularity-project/
├─ app/
│  ├─ dashboard.py
│  └─ pages/
│     └─ 2_Prediction_Lab.py
├─ data/
│  ├─ raw/
│  │  └─ spotify_data_clean.csv
│  └─ processed/
│     └─ spotify_data_processed.csv
├─ notebooks/
├─ outputs/
│  └─ figures/
├─ reports/
├─ src/
│  └─ data_prep.py
├─ run_prep.py
├─ requirements.txt
└─ README.md