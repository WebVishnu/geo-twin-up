from __future__ import annotations

import pandas as pd

from app import data


def temp_frame(district_ids: list[int], years: int = 35) -> pd.DataFrame:
    d = pd.date_range("1991-01-01", periods=365 * years, freq="D")
    return pd.concat([pd.DataFrame({"district_id": i, "date": d, "tmax": 30.0, "tmin": 20.0}) for i in district_ids])


def rain_series(years: int = 35) -> pd.Series:
    d = pd.date_range("1991-01-01", periods=365 * years, freq="D")
    return pd.Series(1.0, index=d)


def test_up_average_needs_every_district() -> None:
    partial = temp_frame(list(range(11)))
    assert not data.temp_ready(partial, rain_series(), None)  # 11 of 75: would be a biased "UP average"
    assert data.temp_ready(temp_frame(list(range(75))), rain_series(), None)


def test_single_district_only_needs_itself() -> None:
    partial = temp_frame([3, 4])
    assert data.temp_ready(partial, rain_series(), 3)
    assert not data.temp_ready(partial, rain_series(), 50)  # not downloaded


def test_short_history_is_not_ready() -> None:
    assert not data.temp_ready(temp_frame([3], years=10), rain_series(), 3)


def test_empty_is_not_ready() -> None:
    assert not data.temp_ready(pd.DataFrame(columns=["district_id", "date", "tmax", "tmin"]), rain_series(), None)
