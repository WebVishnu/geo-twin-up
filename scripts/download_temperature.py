"""Download district temperature (ERA5-Land via Open-Meteo), resumable.

Usage:  python scripts/download_temperature.py 1991 2025
Re-run daily until it reports complete; then it writes data/processed/district_temp_all.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from src.ingest import temperature as T

if __name__ == "__main__":
    start, end = int(sys.argv[1]), int(sys.argv[2])
    g = gpd.read_file("data/external/up_districts.geojson")
    rp = g.geometry.representative_point()
    pts = g[["district_id", "district"]].assign(lat=rp.y.to_numpy(), lon=rp.x.to_numpy())
    chunks = T.plan_chunks(pts, start, end)
    print(f"{len(chunks)} chunks planned", flush=True)
    rep = T.download(chunks)
    print(f"done={rep.done} skipped={rep.skipped} failed={rep.failed} stopped={rep.stopped}", flush=True)
    out = T.combine(names=pts)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/processed/district_temp_all.parquet", index=False)
    print(f"wrote district_temp_all.parquet ({len(out):,} rows). complete={rep.complete}")
