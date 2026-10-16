"""Daily Open-Meteo forecast logger for Uttar Pradesh (ticket T18).

Appends one Parquet file per issue date to data/raw/forecast_log/. The forecast
archive is short-lived upstream, so every day of logging is training data for the
NWP-correction (MOS) model in T10.

Run:  python -m src.ingest.forecast_logger [--force] [--step 0.25]
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"]
# Bounding box of Uttar Pradesh (clipped to the state polygon later, in T5).
UP_BBOX: dict[str, float] = {"lat_min": 23.8, "lat_max": 30.4, "lon_min": 77.1, "lon_max": 84.7}
DEFAULT_OUT = Path("data/raw/forecast_log")
BATCH = 50
FORECAST_DAYS = 8  # lead 0..7


def make_grid(step: float = 0.25, bbox: dict[str, float] = UP_BBOX) -> list[tuple[float, float]]:
    lats = np.round(np.arange(bbox["lat_min"], bbox["lat_max"] + 1e-9, step), 4)
    lons = np.round(np.arange(bbox["lon_min"], bbox["lon_max"] + 1e-9, step), 4)
    return [(float(a), float(o)) for a in lats for o in lons]


def fetch_batch(points: list[tuple[float, float]], session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Call Open-Meteo for up to BATCH points; always returns a list of per-point dicts."""
    s = session or requests
    params: dict[str, str | int | float] = {
        "latitude": ",".join(str(p[0]) for p in points),
        "longitude": ",".join(str(p[1]) for p in points),
        "daily": ",".join(DAILY_VARS),
        "forecast_days": FORECAST_DAYS,
        "timezone": "Asia/Kolkata",
    }
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            r = s.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else [data]
        except Exception as e:  # noqa: BLE001 - retry on any network/HTTP error
            last_err = e
            time.sleep(2**attempt * 2)
    raise RuntimeError(f"Open-Meteo request failed after retries: {last_err}")


def to_frame(
    results: list[dict[str, Any]], issue_date: dt.date, points: list[tuple[float, float]] | None = None
) -> pd.DataFrame:
    """`points` are the requested grid coords (same order as results). Open-Meteo snaps
    returned lat/lon to its own model grid, so we keep the requested ones as `grid_lat/grid_lon`
    for joining with IMD cells."""
    rows = []
    for i, res in enumerate(results):
        d = res["daily"]
        df = pd.DataFrame(
            {
                "valid_date": pd.to_datetime(d["time"]).date,
                "precip": d["precipitation_sum"],
                "tmax": d["temperature_2m_max"],
                "tmin": d["temperature_2m_min"],
            }
        )
        df["lat"] = res["latitude"]
        df["lon"] = res["longitude"]
        df["grid_lat"], df["grid_lon"] = points[i] if points else (res["latitude"], res["longitude"])
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["issue_date"] = issue_date
    out["lead_day"] = (pd.to_datetime(out["valid_date"]) - pd.Timestamp(issue_date)).dt.days
    out["source"] = "open-meteo"
    return out[
        [
            "issue_date",
            "lead_day",
            "valid_date",
            "grid_lat",
            "grid_lon",
            "lat",
            "lon",
            "precip",
            "tmax",
            "tmin",
            "source",
        ]
    ]


def run(
    out_dir: Path = DEFAULT_OUT,
    step: float = 0.25,
    force: bool = False,
    fetch: Callable[[list[tuple[float, float]]], list[dict[str, Any]]] = fetch_batch,
    today: dt.date | None = None,
    sleep_s: float = 0.5,
) -> Path | None:
    """Fetch and store today's forecasts. Idempotent: skips if today's file exists."""
    issue_date = today or dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{issue_date.isoformat()}.parquet"
    if target.exists() and not force:
        return None
    pts = make_grid(step)
    results: list[dict[str, Any]] = []
    for i in range(0, len(pts), BATCH):
        results.extend(fetch(pts[i : i + BATCH]))
        if sleep_s:
            time.sleep(sleep_s)
    df = to_frame(results, issue_date, pts)
    tmp = target.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)  # atomic: no half-written files
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    p = run(Path(a.out), a.step, a.force)
    print(f"wrote {p}" if p else "already logged today; nothing to do")


if __name__ == "__main__":
    main()
