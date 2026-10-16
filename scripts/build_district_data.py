"""Aggregate every cached IMD year to district-level daily rain (run after scripts/download_imd.py).

Usage:  python scripts/build_district_data.py
Writes: data/processed/district_rain_all.parquet  (replaces per-year files in the dashboard)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd

from src.geo import zonal
from src.ingest import imd

RAW = Path("data/raw/imd")
OUT = Path("data/processed/district_rain_all.parquet")


def cached_years() -> list[int]:
    return sorted(int(m.group(1)) for p in (RAW / "rain").glob("*.grd") if (m := re.fullmatch(r"(\d{4})", p.stem)))


if __name__ == "__main__":
    years = cached_years()
    if not years:
        raise SystemExit("No cached IMD years. Run scripts/download_imd.py first.")
    districts = gpd.read_file("data/external/up_districts.geojson")
    parts, weights = [], None
    for y in years:
        da = imd.load_rain(y, y, RAW)
        if weights is None:
            weights = zonal.build_weights(districts, da["lat"].values, da["lon"].values)
        parts.append(zonal.district_series(da, districts, W=weights, name="rain_mm"))
        print(y, "ok", flush=True)
    out = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} rows, years {years[0]}-{years[-1]} ({len(years)} years)")
