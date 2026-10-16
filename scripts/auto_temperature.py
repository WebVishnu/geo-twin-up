"""Unattended temperature download: retry across Open-Meteo's hourly limit, then publish the result.

    python scripts/auto_temperature.py [--push] [--wait-min 62] [--max-rounds 8]

Loop: download -> if an hourly/daily limit stops it, sleep ~1 hour and resume (finished chunks are skipped).
When all chunks exist: write data/processed/district_temp_all.parquet, export app_data/, run the tests, and only
if they pass (and --push was given) commit and push app_data/ ONLY. Anything else in the repo is never touched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from src.ingest import temperature as T

ROOT = Path(__file__).resolve().parent.parent


def run_until_complete(
    attempt: Callable[[], T.DownloadReport],
    sleep: Callable[[float], None] = time.sleep,
    wait_s: float = 62 * 60,
    max_rounds: int = 8,
    log: Callable[[str], None] = print,
) -> bool:
    """Call `attempt` until it reports completion. Returns True if complete, False if rounds ran out."""
    for rnd in range(1, max_rounds + 1):
        rep = attempt()
        log(f"round {rnd}: done={rep.done} skipped={rep.skipped} failed={rep.failed} stopped={rep.stopped}")
        if rep.complete:
            return True
        if rnd == max_rounds:
            break
        why = "usage limit" if rep.stopped else "failed chunks"
        resume = dt.datetime.now() + dt.timedelta(seconds=wait_s)
        log(f"not complete ({why}); sleeping {wait_s / 60:.0f} min, resuming about {resume:%H:%M}")
        sleep(wait_s)
    return False


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def publish() -> bool:
    """Export app_data, run tests, commit and push app_data only. Returns True if pushed."""
    subprocess.run([sys.executable, "scripts/export_app_data.py"], cwd=ROOT, check=True)
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit", "-q"], cwd=ROOT, capture_output=True, text=True
    )
    print(tests.stdout[-400:])
    if tests.returncode != 0:
        print("tests FAILED: not pushing")
        return False
    _git("add", "app_data")
    if _git("diff", "--cached", "--quiet").returncode == 0:
        print("app_data unchanged: nothing to push")
        return False
    msg = "data: district temperature (ERA5-Land, all 75 districts, 1991-2025)\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
    print(_git("commit", "-q", "-m", msg).stderr)
    print(_git("pull", "--rebase", "--autostash", "-q").stderr)
    r = _git("push")
    print(r.stderr.strip())
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1991)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--wait-min", type=float, default=62.0)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--push", action="store_true", help="commit and push app_data/ when complete and tests pass")
    a = ap.parse_args()

    g = gpd.read_file(ROOT / "data/external/up_districts.geojson")
    rp = g.geometry.representative_point()
    pts = g[["district_id", "district"]].assign(lat=rp.y.to_numpy(), lon=rp.x.to_numpy())
    chunks = T.plan_chunks(pts, a.start, a.end)
    root = ROOT / "data/raw/temperature"
    print(f"{len(chunks)} chunks planned; {len(list(root.glob('*.parquet')))} already on disk", flush=True)

    ok = run_until_complete(lambda: T.download(chunks, root), wait_s=a.wait_min * 60, max_rounds=a.max_rounds)
    out = T.combine(root, pts)
    (ROOT / "data/processed").mkdir(exist_ok=True)
    out.to_parquet(ROOT / "data/processed/district_temp_all.parquet", index=False)
    print(f"combined: {len(out):,} rows, {out['district_id'].nunique()} districts", flush=True)
    if not ok:
        print("INCOMPLETE after max rounds; run again later. Nothing was published.", flush=True)
        return 1
    if a.push:
        publish()
    else:
        print("complete. Run scripts/export_app_data.py and push, or re-run with --push.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
