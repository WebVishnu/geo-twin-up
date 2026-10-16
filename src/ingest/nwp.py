"""Lead-specific NWP rainfall forecasts from Open-Meteo's Previous Runs API (ticket T10).

`precipitation_previous_dayK` at a valid hour is what the model run issued K days earlier predicted for
that hour. Summed over a local day it gives "the K-day-ahead forecast for that date", which is exactly what
a MOS (model output statistics) correction needs for training. Daily previous-day variables do not exist,
so we download hourly values and sum them ourselves. Coverage starts in 2024 (checked in docs/data_findings.md).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src.ingest.temperature import RateLimitedError, _scope

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DEFAULT_ROOT = Path("data/raw/nwp")
DEFAULT_LEADS = (1, 3, 7)


@dataclass(frozen=True)
class NwpChunk:
    district_id: int
    lat: float
    lon: float
    start: str  # ISO date
    end: str

    @property
    def name(self) -> str:
        return f"{self.district_id:03d}_{self.start}_{self.end}.parquet"


@dataclass
class NwpReport:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    stopped: str | None = None

    @property
    def complete(self) -> bool:
        return self.stopped is None and self.failed == 0


Fetcher = Callable[[float, float, str, str, tuple[int, ...]], pd.DataFrame]


def hourly_to_daily(times: list[str], series: dict[int, list[float | None]]) -> pd.DataFrame:
    """Sum hourly forecasts to local days. Days with any missing hour are dropped (never a partial sum)."""
    frames = []
    ts = pd.to_datetime(pd.Series(times))
    day = ts.dt.normalize()
    for lead, vals in series.items():
        s = pd.Series(vals, dtype="float64")
        g = pd.DataFrame({"valid_date": day, "v": s}).groupby("valid_date")["v"]
        total = g.sum(min_count=24)  # NaN unless all 24 hours are present
        frames.append(pd.DataFrame({"valid_date": total.index, "lead": lead, "nwp_precip": total.to_numpy()}))
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["nwp_precip"]).reset_index(drop=True)


def fetch_chunk(lat: float, lon: float, start: str, end: str, leads: tuple[int, ...] = DEFAULT_LEADS) -> pd.DataFrame:
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(f"precipitation_previous_day{k}" for k in leads),
        "timezone": "Asia/Kolkata",
    }
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = requests.get(API_URL, params=params, timeout=90)
            if r.status_code == 429 or (not r.ok and "limit" in r.text[:300].lower()):
                raise RateLimitedError(r.text[:200], _scope(r.text[:300]))
            r.raise_for_status()
            h = r.json()["hourly"]
            return hourly_to_daily(h["time"], {k: h[f"precipitation_previous_day{k}"] for k in leads})
        except RateLimitedError:
            raise
        except (requests.RequestException, KeyError, ValueError) as e:
            last = e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"nwp fetch failed after retries: {last}")


def plan_chunks(points: pd.DataFrame, start: str, end: str) -> list[NwpChunk]:
    """One chunk per district per calendar year in [start, end]. points: district_id, lat, lon."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    spans = [
        (max(s, pd.Timestamp(y, 1, 1)).date().isoformat(), min(e, pd.Timestamp(y, 12, 31)).date().isoformat())
        for y in range(s.year, e.year + 1)
    ]
    return [
        NwpChunk(int(str(r.district_id)), float(str(r.lat)), float(str(r.lon)), a, b)
        for r in points.itertuples(index=False)
        for a, b in spans
    ]


def download(
    chunks: list[NwpChunk],
    root: Path = DEFAULT_ROOT,
    leads: tuple[int, ...] = DEFAULT_LEADS,
    fetch: Fetcher = fetch_chunk,
    sleep_s: float = 1.0,
    minute_wait_s: float = 65.0,
    minute_retries: int = 5,
    wait: Callable[[float], None] = time.sleep,
) -> NwpReport:
    """Resumable download. Per-minute limits are waited out; hourly/daily limits stop cleanly."""
    root.mkdir(parents=True, exist_ok=True)
    rep = NwpReport()
    for c in chunks:
        target = root / c.name
        if target.exists():
            rep.skipped += 1
            continue
        df: pd.DataFrame | None = None
        for attempt in range(minute_retries + 1):
            try:
                df = fetch(c.lat, c.lon, c.start, c.end, leads)
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


def combine(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """[district_id, valid_date, lead, nwp_precip]; later chunks win on overlap."""
    files = sorted(root.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["district_id", "valid_date", "lead", "nwp_precip"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates(["district_id", "valid_date", "lead"], keep="last").reset_index(drop=True)
