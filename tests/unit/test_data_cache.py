from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from app import data


def write_rain(path: Path, n_days: int, mtime: float) -> None:
    d = pd.date_range("2000-01-01", periods=n_days)
    pd.DataFrame({"district_id": 0, "date": d, "rain_mm": 1.0}).astype({"district_id": "int16"}).to_parquet(path)
    os.utime(path, (mtime, mtime))


def test_refreshed_data_file_is_not_served_from_a_stale_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a redeploy with new app_data/rain.parquet kept showing the old table (11 years vs 35)."""
    monkeypatch.setattr(data, "APP_DATA", tmp_path)
    f = tmp_path / "rain.parquet"
    write_rain(f, 10, 1_700_000_000)
    assert len(data.load_rain()) == 10
    write_rain(f, 25, 1_700_000_500)  # same path, new content, new modified time
    assert len(data.load_rain()) == 25


def test_missing_file_returns_empty_not_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "APP_DATA", tmp_path)
    assert data.load_rain().empty and data.load_temp().empty
    fc, issued = data.load_forecast()
    assert fc.empty and issued is None
