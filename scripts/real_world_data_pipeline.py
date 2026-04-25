from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
MUSICBRAINZ_BASE = "https://musicbrainz.org/ws/2/recording"
DEFAULT_DB = "data/live/music_live_monitor.sqlite"
DEFAULT_CSV = "data/live/lastfm_top_tracks_snapshots.csv"
USER_AGENT = "spotify-real-world-project/1.0 (student-project-demo)"


@dataclass
class TrackSnapshot:
    snapshot_ts_utc: str
    snapshot_date: str
    chart_source: str
    chart_page: int
    chart_rank: int
    track_name: str
    artist_name: str
    lastfm_url: Optional[str]
    lastfm_mbid: Optional[str]
    lastfm_duration_ms: Optional[float]
    lastfm_listeners: Optional[float]
    lastfm_playcount: Optional[float]
    musicbrainz_recording_id: Optional[str]
    musicbrainz_release_title: Optional[str]
    musicbrainz_release_date: Optional[str]
    musicbrainz_primary_type: Optional[str]
    musicbrainz_first_release_year: Optional[float]
    track_key: str


class ApiClient:
    def __init__(self, timeout: int = 20) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


def normalize_key(artist: str, track: str) -> str:
    return f"{artist.strip().lower()}|||{track.strip().lower()}"


def parse_int(value: Any) -> Optional[float]:
    if value in (None, "", []):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_lastfm_chart(api: ApiClient, api_key: str, page: int = 1, limit: int = 100) -> list[dict[str, Any]]:
    payload = api.get_json(
        LASTFM_BASE,
        {
            "method": "chart.gettoptracks",
            "api_key": api_key,
            "format": "json",
            "page": page,
            "limit": limit,
        },
    )
    tracks = payload.get("tracks", {}).get("track", [])
    if isinstance(tracks, dict):
        tracks = [tracks]
    return tracks


def fetch_lastfm_track_info(api: ApiClient, api_key: str, artist: str, track: str) -> dict[str, Any]:
    payload = api.get_json(
        LASTFM_BASE,
        {
            "method": "track.getInfo",
            "api_key": api_key,
            "artist": artist,
            "track": track,
            "autocorrect": 1,
            "format": "json",
        },
    )
    return payload.get("track", {})


def fetch_musicbrainz_recording(api: ApiClient, artist: str, track: str, sleep_seconds: float = 1.0) -> dict[str, Any]:
    """MusicBrainz asks clients to identify themselves and avoid aggressive request rates.
    We keep a simple 1-second sleep between requests.
    """
    params = {
        "query": f'recording:"{track}" AND artist:"{artist}"',
        "fmt": "json",
        "limit": 1,
    }
    payload = api.get_json(MUSICBRAINZ_BASE, params)
    time.sleep(sleep_seconds)
    recordings = payload.get("recordings", [])
    return recordings[0] if recordings else {}


def extract_musicbrainz_fields(recording: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[float]]:
    if not recording:
        return None, None, None, None, None

    recording_id = recording.get("id")
    first_release_date = recording.get("first-release-date")
    first_release_year = None
    if isinstance(first_release_date, str) and len(first_release_date) >= 4 and first_release_date[:4].isdigit():
        first_release_year = float(first_release_date[:4])

    release_title = None
    release_date = None
    primary_type = None

    releases = recording.get("releases", []) or []
    if releases:
        first_release = releases[0]
        release_title = first_release.get("title")
        release_date = first_release.get("date")
        group = first_release.get("release-group", {}) or {}
        primary_type = group.get("primary-type")

    return recording_id, release_title, release_date or first_release_date, primary_type, first_release_year


def build_snapshots(api_key: str, page: int, limit: int) -> pd.DataFrame:
    api = ApiClient()
    chart_rows = fetch_lastfm_chart(api, api_key=api_key, page=page, limit=limit)

    now = datetime.now(timezone.utc)
    snapshot_ts = now.isoformat()
    snapshot_date = now.date().isoformat()

    records: list[TrackSnapshot] = []
    for idx, row in enumerate(chart_rows, start=1):
        track_name = row.get("name")
        artist_name = (row.get("artist") or {}).get("name")
        if not track_name or not artist_name:
            continue

        try:
            track_info = fetch_lastfm_track_info(api, api_key=api_key, artist=artist_name, track=track_name)
        except requests.HTTPError:
            track_info = {}

        try:
            mb_recording = fetch_musicbrainz_recording(api, artist=artist_name, track=track_name)
        except requests.HTTPError:
            mb_recording = {}

        mb_recording_id, mb_release_title, mb_release_date, mb_primary_type, mb_year = extract_musicbrainz_fields(mb_recording)

        record = TrackSnapshot(
            snapshot_ts_utc=snapshot_ts,
            snapshot_date=snapshot_date,
            chart_source="lastfm_chart_toptracks",
            chart_page=page,
            chart_rank=idx,
            track_name=track_name,
            artist_name=artist_name,
            lastfm_url=track_info.get("url") or row.get("url"),
            lastfm_mbid=track_info.get("mbid") or row.get("mbid"),
            lastfm_duration_ms=parse_int(track_info.get("duration")),
            lastfm_listeners=parse_int(track_info.get("listeners") or row.get("listeners")),
            lastfm_playcount=parse_int(track_info.get("playcount") or row.get("playcount")),
            musicbrainz_recording_id=mb_recording_id,
            musicbrainz_release_title=mb_release_title,
            musicbrainz_release_date=mb_release_date,
            musicbrainz_primary_type=mb_primary_type,
            musicbrainz_first_release_year=mb_year,
            track_key=normalize_key(artist_name, track_name),
        )
        records.append(record)

    return pd.DataFrame([asdict(r) for r in records])


def append_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str = "lastfm_track_snapshots") -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="append", index=False)


def append_to_csv(df: pd.DataFrame, csv_path: str) -> None:
    output = Path(csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    df.to_csv(output, mode="a", header=write_header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live music chart data from Last.fm and MusicBrainz.")
    parser.add_argument("--lastfm-api-key", default=os.getenv("LASTFM_API_KEY"))
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--db-path", default=DEFAULT_DB)
    parser.add_argument("--csv-path", default=DEFAULT_CSV)
    parser.add_argument("--print-head", action="store_true")
    args = parser.parse_args()

    if not args.lastfm_api_key:
        raise SystemExit("LASTFM_API_KEY is required. Pass --lastfm-api-key or set the environment variable.")

    df = build_snapshots(api_key=args.lastfm_api_key, page=args.page, limit=args.limit)
    if df.empty:
        raise SystemExit("No live rows were collected.")

    append_to_sqlite(df, db_path=args.db_path)
    append_to_csv(df, csv_path=args.csv_path)

    latest_path = Path(args.csv_path).with_name("lastfm_top_tracks_latest.csv")
    df.to_csv(latest_path, index=False)

    manifest = {
        "rows": int(len(df)),
        "snapshot_date": str(df["snapshot_date"].iloc[0]),
        "csv_path": str(args.csv_path),
        "db_path": str(args.db_path),
        "latest_path": str(latest_path),
    }
    manifest_path = Path(args.csv_path).with_name("lastfm_collection_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    if args.print_head:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
