"""Export the small, deployable data files the dashboard reads (app_data/).

The deployed app never touches raw grids or downloads anything. Usage:
    python scripts/export_app_data.py                 # rain + temperature + latest forecast
    python scripts/export_app_data.py --forecast-only # what the daily GitHub Action runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd

from src.geo.forecast_agg import aggregate_forecast

OUT = Path("app_data")
GEO = Path("data/external/up_districts.geojson")
FORECAST_DIR = Path("data/raw/forecast_log")


def export_forecast() -> None:
    files = sorted(FORECAST_DIR.glob("*.parquet"))
    if not files:
        print("no forecast logs; skipped")
        return
    fc = pd.read_parquet(files[-1])
    agg = aggregate_forecast(fc, gpd.read_file(GEO))
    agg["issue_date"] = pd.to_datetime(fc["issue_date"].iloc[0])
    agg = agg.astype(
        {"district_id": "int16", "lead_day": "int8", "precip": "float32", "tmax": "float32", "tmin": "float32"}
    )
    agg.to_parquet(OUT / "forecast.parquet", index=False, compression="zstd")
    print(f"forecast.parquet: {len(agg)} rows, issued {agg['issue_date'].iloc[0].date()}")


def export_series(src: Path, dst: str, cols: list[str]) -> None:
    if not src.exists():
        print(f"{src} missing; skipped {dst}")
        return
    df = pd.read_parquet(src, columns=["district_id", "date", *cols]).drop_duplicates(["district_id", "date"])
    df = df.sort_values(["district_id", "date"]).astype({"district_id": "int16", **{c: "float32" for c in cols}})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df.to_parquet(OUT / dst, index=False, compression="zstd")
    yrs = df["date"].dt.year
    print(f"{dst}: {len(df):,} rows, {yrs.min()}-{yrs.max()}, {(OUT / dst).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    export_forecast()
    if not a.forecast_only:
        export_series(Path("data/processed/district_rain_all.parquet"), "rain.parquet", ["rain_mm"])
        export_series(Path("data/processed/district_temp_all.parquet"), "temp.parquet", ["tmax", "tmin"])
