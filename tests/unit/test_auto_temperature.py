from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from src.ingest.temperature import DownloadReport

_spec = importlib.util.spec_from_file_location(
    "auto_temperature", Path(__file__).resolve().parents[2] / "scripts" / "auto_temperature.py"
)
assert _spec and _spec.loader
auto = importlib.util.module_from_spec(_spec)
sys.modules["auto_temperature"] = auto
_spec.loader.exec_module(auto)


def reports(*items: DownloadReport):
    it = iter(items)
    return lambda: next(it)


def test_waits_one_hour_between_limited_rounds_then_finishes() -> None:
    waits: list[float] = []
    ok = auto.run_until_complete(
        reports(
            DownloadReport(done=100, stopped="limit"), DownloadReport(done=50, stopped="limit"), DownloadReport(done=40)
        ),
        sleep=waits.append,
        wait_s=3720,
        log=lambda s: None,
    )
    assert ok and waits == [3720, 3720]


def test_complete_on_first_round_never_sleeps() -> None:
    waits: list[float] = []
    assert auto.run_until_complete(reports(DownloadReport(done=5)), sleep=waits.append, log=lambda s: None)
    assert waits == []


def test_gives_up_after_max_rounds_without_a_final_sleep() -> None:
    waits: list[float] = []
    ok = auto.run_until_complete(
        reports(*[DownloadReport(stopped="limit")] * 3), sleep=waits.append, max_rounds=3, log=lambda s: None
    )
    assert not ok and len(waits) == 2


def test_failed_chunks_are_retried_too() -> None:
    waits: list[float] = []
    ok = auto.run_until_complete(
        reports(DownloadReport(done=10, failed=2), DownloadReport(done=2)), sleep=waits.append, log=lambda s: None
    )
    assert ok and len(waits) == 1
