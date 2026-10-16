"""District daily temperature from Open-Meteo's ERA5-Land archive (0.1 deg).

IMD's temperature archive is 1 deg (about 100 km), too coarse for district maps, so district
temperature comes from ERA5-Land sampled at each district's representative point. This is a
reanalysis value at one point, not an area mean and not an IMD observation: say so wherever it is shown.

The free tier limits daily usage, so the download is chunked per (district, 5 years), cached on disk,
and resumable: when the API says "limit reached" we stop cleanly and the next run continues.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5_land"
DEFAULT_ROOT = Path("data/raw/temperature")


class RateLimitedError(RuntimeError):
    """Raised when Open-Meteo reports that a usage limit is reached.

    `scope` is "minute" (wait about a minute and retry) or "other" (hourly/daily: stop and resume later).
    """

    def __init__(self, message: str, scope: str = "other") -> None:
        super().__init__(message)
        self.scope = scope


def _scope(text: str) -> str:
    return "minute" if "minutely" in text.lower() or "one minute" in text.lower() else "other"


@dataclass(frozen=True)
class Chunk:
    """One download unit: a district over a block of years."""

    district_id: int
    lat: float
    lon: float
    start_year: int
    end_year: int

    @property
    def name(self) -> str:
        return f"{self.district_id:03d}_{self.start_year}-{self.end_year}.parquet"


@dataclass
class DownloadReport:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    stopped: str | None = None

    @property
    def complete(self) -> bool:
        return self.stopped is None and self.failed == 0


Fetcher = Callable[[float, float, str, str], pd.DataFrame]


def plan_chunks(points: pd.DataFrame, start_year: int, end_year: int, years_per_chunk: int = 5) -> list[Chunk]:
    """points: columns district_id, lat, lon."""
    spans = [(y, min(y + years_per_chunk - 1, end_year)) for y in range(start_year, end_year + 1, years_per_chunk)]
    return [
        Chunk(int(str(r.district_id)), float(str(r.lat)), float(str(r.lon)), a, b)
        for r in points.itertuples(index=False)
        for a, b in spans
    ]


def fetch_chunk(lat: float, lon: float, start: str, end: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch daily tmax/tmin for one point. Raises RateLimitedError on quota errors."""
    s = session or requests
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min",
        "models": MODEL,
        "timezone": "Asia/Kolkata",
    }
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = s.get(API_URL, params=params, timeout=60)
            if r.status_code == 429 or (not r.ok and "limit" in r.text[:300].lower()):
                raise RateLimitedError(r.text[:200], _scope(r.text[:300]))
            r.raise_for_status()
            d = r.json()["daily"]
            return pd.DataFrame(
                {"date": pd.to_datetime(d["time"]), "tmax": d["temperature_2m_max"], "tmin": d["temperature_2m_min"]}
            )
        except RateLimitedError:
            raise
        except (requests.RequestException, KeyError, ValueError) as e:
            last = e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"temperature fetch failed after retries: {last}")


def download(
    chunks: list[Chunk],
    root: Path = DEFAULT_ROOT,
    fetch: Fetcher = fetch_chunk,
    sleep_s: float = 1.0,
    minute_wait_s: float = 65.0,
    minute_retries: int = 5,
    wait: Callable[[float], None] = time.sleep,
) -> DownloadReport:
    """Download missing chunks. Per-minute limits are waited out; hourly/daily limits stop cleanly.

    Safe to re-run: finished chunks are skipped.
    """
    root.mkdir(parents=True, exist_ok=True)
    rep = DownloadReport()
    for c in chunks:
        target = root / c.name
        if target.exists():
            rep.skipped += 1
            continue
        df: pd.DataFrame | None = None
        for attempt in range(minute_retries + 1):
            try:
                df = fetch(c.lat, c.lon, f"{c.start_year}-01-01", f"{c.end_year}-12-31")
                break
            except RateLimitedError as e:
                if e.scope == "minute" and attempt < minute_retries:
                    wait(minute_wait_s)
                    continue
                rep.stopped = f"rate limited at {c.name}: {e}"
                break
            except RuntimeError:
                rep.failed += 1
                break
        if rep.stopped:
            break
        if df is None:
            continue
        df.insert(0, "district_id", c.district_id)
        tmp = target.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(target)
        rep.done += 1
        if sleep_s:
            time.sleep(sleep_s)
    return rep


def combine(root: Path = DEFAULT_ROOT, names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Concatenate cached chunks into [district_id, (district), date, tmax, tmin]."""
    files = sorted(root.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["district_id", "date", "tmax", "tmin"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["district_id", "date"]).sort_values(["district_id", "date"]).reset_index(drop=True)
    if names is not None:
        df = df.merge(names[["district_id", "district"]], on="district_id", how="left")
    return df
