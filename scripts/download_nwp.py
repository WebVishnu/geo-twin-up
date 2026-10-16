"""Download lead-specific NWP rain (Open-Meteo Previous Runs) for every district centre. Resumable.

Usage:  python scripts/download_nwp.py            (2024-01-01 .. 3 days ago)
Writes: data/raw/nwp/*.parquet, then data/processed/district_nwp.parquet
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from src.ingest import nwp

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else (dt.date.today() - dt.timedelta(days=3)).isoformat()
    g = gpd.read_file("data/external/up_districts.geojson")
    rp = g.geometry.representative_point()
    pts = g[["district_id"]].assign(lat=rp.y.to_numpy(), lon=rp.x.to_numpy())
    chunks = nwp.plan_chunks(pts, start, end)
    print(f"{len(chunks)} chunks planned ({start}..{end})", flush=True)
    rep = nwp.download(chunks)
    print(f"done={rep.done} skipped={rep.skipped} failed={rep.failed} stopped={rep.stopped}", flush=True)
    out = nwp.combine()
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/processed/district_nwp.parquet", index=False)
    print(f"wrote district_nwp.parquet ({len(out):,} rows). complete={rep.complete}")
