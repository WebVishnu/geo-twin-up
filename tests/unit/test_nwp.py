from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest import nwp
from src.ingest.temperature import RateLimitedError


def hourly(days: int = 3, lead_vals: dict[int, float] | None = None) -> tuple[list[str], dict[int, list[float | None]]]:
    times = [str(t) for t in pd.date_range("2024-07-01", periods=24 * days, freq="h")]
    lv = lead_vals or {1: 0.5, 3: 1.0}
    return times, {k: [v] * len(times) for k, v in lv.items()}


def test_hourly_sums_to_local_days() -> None:
    t, s = hourly()
    d = nwp.hourly_to_daily(t, s)
    assert set(d.lead) == {1, 3} and len(d) == 6
    assert np.allclose(d[d.lead == 1].nwp_precip, 12.0) and np.allclose(d[d.lead == 3].nwp_precip, 24.0)


def test_partial_days_are_dropped_not_summed() -> None:
    t, s = hourly(days=2)
    s[1][5] = None  # one missing hour on day 1 for lead 1
    d = nwp.hourly_to_daily(t, s)
    assert len(d[d.lead == 1]) == 1 and len(d[d.lead == 3]) == 2


def points() -> pd.DataFrame:
    return pd.DataFrame({"district_id": [0, 1], "lat": 26.0, "lon": 80.0})


def test_plan_chunks_split_by_year() -> None:
    ch = nwp.plan_chunks(points(), "2024-01-01", "2025-10-10")
    assert len(ch) == 4 and ch[0].start == "2024-01-01" and ch[1].end == "2025-10-10"


def fake(lat: float, lon: float, start: str, end: str, leads: tuple[int, ...]) -> pd.DataFrame:
    d = pd.date_range(start, end)
    return pd.concat(
        [pd.DataFrame({"valid_date": d, "lead": k, "nwp_precip": 1.0 * k}) for k in leads], ignore_index=True
    )


def test_download_resumable_and_combine(tmp_path: Path) -> None:
    ch = nwp.plan_chunks(points(), "2024-01-01", "2024-03-31")
    r1 = nwp.download(ch, tmp_path, fetch=fake, sleep_s=0)
    r2 = nwp.download(ch, tmp_path, fetch=fake, sleep_s=0)
    assert r1.complete and r1.done == 2 and r2.skipped == 2 and r2.done == 0
    df = nwp.combine(tmp_path)
    assert df.duplicated(["district_id", "valid_date", "lead"]).sum() == 0 and set(df.lead) == {1, 3, 7}


def test_rate_limit_stops_then_resumes(tmp_path: Path) -> None:
    n = {"c": 0}

    def flaky(lat: float, lon: float, start: str, end: str, leads: tuple[int, ...]) -> pd.DataFrame:
        n["c"] += 1
        if n["c"] == 2:
            raise RateLimitedError("Hourly API request limit exceeded", "other")
        return fake(lat, lon, start, end, leads)

    ch = nwp.plan_chunks(points(), "2024-01-01", "2024-03-31")
    r1 = nwp.download(ch, tmp_path, fetch=flaky, sleep_s=0)
    assert r1.done == 1 and r1.stopped and not r1.complete
    assert nwp.download(ch, tmp_path, fetch=fake, sleep_s=0).complete
