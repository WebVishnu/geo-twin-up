"""Ticket T2: data smoke tests. Prints PASS/FAIL per source and writes docs/data_findings.md.

Usage:  python scripts/smoke_data.py
Credentials (never commit): EARTHDATA_USER / EARTHDATA_PASS env vars for IMERG.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
import traceback
from pathlib import Path

import requests

RESULTS: list[tuple[str, str, str]] = []
LUCKNOW = (26.85, 80.95)


def record(name: str, ok: bool, note: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}")


def check_imdlib() -> None:
    try:
        import imdlib as imd

        tmp = tempfile.mkdtemp()
        year = dt.date.today().year - 2
        data = imd.get_data("rain", year, year, fn_format="yearwise", file_dir=tmp)
        ds = data.get_xarray()
        record("IMD rain archive (imdlib)", True, f"year {year}: dims={dict(ds.sizes)}")
        # probe the most recent complete archive year
        last_ok = None
        for y in range(dt.date.today().year, dt.date.today().year - 4, -1):
            try:
                imd.get_data("rain", y, y, fn_format="yearwise", file_dir=tmp)
                last_ok = y
                break
            except Exception:  # noqa: BLE001
                continue
        record("IMD latest archive year", last_ok is not None, f"latest downloadable year: {last_ok}")
    except Exception as e:  # noqa: BLE001
        record("IMD rain archive (imdlib)", False, f"{type(e).__name__}: {e}")
    try:
        import imdlib as imd

        fn = getattr(imd, "get_real_data", None)
        record("imdlib real-time function present", fn is not None, "get_real_data" if fn else "missing")
    except Exception as e:  # noqa: BLE001
        record("imdlib real-time function present", False, str(e))


def check_open_meteo() -> None:
    lat, lon = LUCKNOW
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum,temperature_2m_max",
                "forecast_days": 8,
                "timezone": "Asia/Kolkata",
            },
            timeout=30,
        )
        r.raise_for_status()
        n = len(r.json()["daily"]["time"])
        record("Open-Meteo forecast", True, f"{n} days returned")
    except Exception as e:  # noqa: BLE001
        record("Open-Meteo forecast", False, str(e))
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "daily": "precipitation_sum",
                "timezone": "Asia/Kolkata",
            },
            timeout=30,
        )
        r.raise_for_status()
        record("Open-Meteo archive (ERA5)", True, f"{len(r.json()['daily']['time'])} days")
    except Exception as e:  # noqa: BLE001
        record("Open-Meteo archive (ERA5)", False, str(e))
    # Previous Runs: how far back does lead-specific data go?
    try:
        r = requests.get(
            "https://previous-runs-api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum_previous_day3",
                "start_date": "2023-01-01",
                "end_date": "2023-01-07",
                "timezone": "Asia/Kolkata",
            },
            timeout=30,
        )
        has = r.ok and any(v is not None for v in r.json().get("daily", {}).get("precipitation_sum_previous_day3", []))
        record(
            "Open-Meteo Previous Runs @2023",
            has,
            "data present in 2023" if has else "no data in 2023 (archive starts later; probe 2024/2025)",
        )
    except Exception as e:  # noqa: BLE001
        record("Open-Meteo Previous Runs @2023", False, str(e))


def check_earthdata() -> None:
    user, pw = os.getenv("EARTHDATA_USER"), os.getenv("EARTHDATA_PASS")
    if not (user and pw):
        record("Earthdata/IMERG", False, "set EARTHDATA_USER and EARTHDATA_PASS env vars to test (skipped)")
        return
    try:
        import earthaccess  # noqa: F401

        record("Earthdata/IMERG", True, "earthaccess installed; run login+search manually (see docs)")
    except Exception as e:  # noqa: BLE001
        record("Earthdata/IMERG", False, f"pip install earthaccess ({e})")


def check_aifs() -> None:
    try:
        r = requests.head("https://data.ecmwf.int/forecasts/", timeout=30, allow_redirects=True)
        record("ECMWF open data endpoint reachable", r.status_code < 400, f"HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001
        record("ECMWF open data endpoint reachable", False, str(e))


def write_doc() -> None:
    Path("docs").mkdir(exist_ok=True)
    lines = [f"# Data findings (smoke test {dt.date.today()})", "", "| Check | Result | Note |", "|---|---|---|"]
    lines += [f"| {n} | {s} | {note} |" for n, s, note in RESULTS]
    lines += [
        "",
        "## Decisions to record",
        "- imdlib real-time lag: ",
        "- NWP training archive length (Previous Runs): ",
        "- MOSDAC application submitted on: ",
        "",
    ]
    Path("docs/data_findings.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote docs/data_findings.md")


if __name__ == "__main__":
    for f in (check_imdlib, check_open_meteo, check_earthdata, check_aifs):
        try:
            f()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    write_doc()
