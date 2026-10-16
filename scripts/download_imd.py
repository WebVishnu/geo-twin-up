"""Patient IMD rain downloader (IMD's server is slow / intermittently times out).

Retries each year with backoff and skips years already cached, so it is safe to
stop and re-run.   Usage:  python scripts/download_imd.py 1991 2025
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import imdlib as imd
import requests

OUT = Path("data/raw/imd")


def fetch_year(year: int, var: str = "rain", tries: int = 12) -> bool:
    # imdlib passes no timeout to requests, so a dead connection can hang; set a default.
    orig = requests.sessions.Session.request

    def with_timeout(self, *a, **k):
        k.setdefault("timeout", (60, 600))
        return orig(self, *a, **k)

    requests.sessions.Session.request = with_timeout
    try:
        for i in range(tries):
            try:
                imd.get_data(var, year, year, fn_format="yearwise", file_dir=str(OUT))
                return True
            except Exception as e:  # noqa: BLE001
                wait = min(30 * (i + 1), 300)
                print(f"{year}: attempt {i + 1}/{tries} failed ({type(e).__name__}); retry in {wait}s", flush=True)
                time.sleep(wait)
        return False
    finally:
        requests.sessions.Session.request = orig


def cached(year: int, var: str = "rain") -> bool:
    return any(OUT.rglob(f"*{year}*.grd")) if OUT.exists() else False


if __name__ == "__main__":
    start, end = int(sys.argv[1]), int(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    for y in range(start, end + 1):
        if cached(y):
            print(f"{y}: cached", flush=True)
            continue
        print(f"{y}: {'OK' if fetch_year(y) else 'GAVE UP'}", flush=True)
