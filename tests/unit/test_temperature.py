from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingest import temperature as T


def points(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"district_id": range(n), "district": [f"D{i}" for i in range(n)], "lat": 26.0, "lon": 80.0})


def fake_fetch(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    d = pd.date_range(start, end)
    return pd.DataFrame({"date": d, "tmax": 35.0, "tmin": 22.0})


def test_plan_chunks_covers_all_years_without_overlap() -> None:
    ch = T.plan_chunks(points(1), 1991, 2025, years_per_chunk=5)
    assert [(c.start_year, c.end_year) for c in ch][0] == (1991, 1995)
    assert [(c.start_year, c.end_year) for c in ch][-1] == (2021, 2025)
    spans = [(c.start_year, c.end_year) for c in ch]
    assert sum(b - a + 1 for a, b in spans) == 35


def test_download_is_resumable_and_idempotent(tmp_path: Path) -> None:
    ch = T.plan_chunks(points(2), 1991, 2000)
    r1 = T.download(ch, tmp_path, fake_fetch, sleep_s=0)
    assert r1.done == len(ch) and r1.complete
    r2 = T.download(ch, tmp_path, fake_fetch, sleep_s=0)
    assert r2.done == 0 and r2.skipped == len(ch)


def test_rate_limit_stops_cleanly_and_next_run_continues(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] == 3:
            raise T.RateLimitedError("Daily API request limit exceeded")
        return fake_fetch(lat, lon, start, end)

    ch = T.plan_chunks(points(2), 1991, 2000)  # 4 chunks
    r1 = T.download(ch, tmp_path, flaky, sleep_s=0)
    assert r1.done == 2 and r1.stopped and not r1.complete
    r2 = T.download(ch, tmp_path, fake_fetch, sleep_s=0)
    assert r2.skipped == 2 and r2.done == 2 and r2.complete


def test_minutely_limit_is_waited_out_not_fatal(tmp_path: Path) -> None:
    calls = {"n": 0}
    waits: list[float] = []

    def minutely(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] in (2, 3):
            raise T.RateLimitedError("Minutely API request limit exceeded", "minute")
        return fake_fetch(lat, lon, start, end)

    ch = T.plan_chunks(points(2), 1991, 2000)  # 4 chunks
    r = T.download(ch, tmp_path, minutely, sleep_s=0, wait=waits.append)
    assert r.complete and r.done == len(ch) and waits == [65.0, 65.0]


def test_minutely_limit_gives_up_after_retries(tmp_path: Path) -> None:
    def always(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        raise T.RateLimitedError("Minutely API request limit exceeded", "minute")

    r = T.download(
        T.plan_chunks(points(1), 1991, 1995), tmp_path, always, sleep_s=0, minute_retries=2, wait=lambda s: None
    )
    assert r.stopped and not r.complete


def test_scope_detection() -> None:
    assert T._scope('{"reason":"Minutely API request limit exceeded. Please try again in one minute."}') == "minute"
    assert T._scope('{"reason":"Daily API request limit exceeded."}') == "other"


def test_hard_failure_is_counted_not_fatal(tmp_path: Path) -> None:
    def broken(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        raise RuntimeError("boom")

    r = T.download(T.plan_chunks(points(1), 1991, 1995), tmp_path, broken, sleep_s=0)
    assert r.failed == 1 and r.done == 0 and not r.complete


def test_combine_dedupes_and_joins_names(tmp_path: Path) -> None:
    ch = T.plan_chunks(points(2), 1991, 1995)
    T.download(ch, tmp_path, fake_fetch, sleep_s=0)
    df = T.combine(tmp_path, points(2))
    assert len(df) == 2 * 1826 and df.district.notna().all()
    assert df.duplicated(["district_id", "date"]).sum() == 0


def test_combine_empty(tmp_path: Path) -> None:
    assert T.combine(tmp_path).empty


def test_no_partial_files_left_behind(tmp_path: Path) -> None:
    T.download(T.plan_chunks(points(1), 1991, 1995), tmp_path, fake_fetch, sleep_s=0)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("sy,ey", [(1991, 1991), (1991, 1993)])
def test_short_ranges(sy: int, ey: int) -> None:
    ch = T.plan_chunks(points(1), sy, ey)
    assert len(ch) == 1 and ch[0].end_year == ey
