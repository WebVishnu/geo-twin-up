from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import baselines, metrics, splits
from src.indices import hazards, spi


def test_corr_nan_for_constant_forecast() -> None:
    assert np.isnan(metrics.corr([1, 2, 3], [5, 5, 5]))
    assert metrics.corr([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_brier_and_skill() -> None:
    y = np.array([0.0, 5.0, 0.0, 5.0])
    assert metrics.brier(y, np.array([0, 1, 0, 1.0]), 2.5) == 0.0
    assert metrics.brier(y, np.full(4, 0.5), 2.5) == pytest.approx(0.25)


def test_table_has_skill_vs_climatology() -> None:
    y = np.array([0, 0, 4, 20.0, 0, 3])
    t = metrics.table(y, {"climatology": np.full(6, y.mean()), "perfect": y})
    assert t.set_index("forecast").loc["perfect", "rmse_skill_vs_clim"] == pytest.approx(1.0)
    assert t.set_index("forecast").loc["climatology", "rmse_skill_vs_clim"] == pytest.approx(0.0)


def test_categorical_nan_when_no_events() -> None:
    c = metrics.categorical(np.zeros(3), np.zeros(3), 2.5)
    assert np.isnan(c["pod"]) and np.isnan(c["csi"])


def test_climatology_forecast_lookup() -> None:
    idx = pd.date_range("2019-01-01", "2021-12-31")
    wide = pd.DataFrame({0: np.where(idx.month == 7, 10.0, 0.0)}, index=idx)
    clim = baselines.fit_climatology(wide, [2019, 2020, 2021])
    feats = pd.DataFrame({"district_id": [0, 0], "target_date": pd.to_datetime(["2022-07-15", "2022-01-15"])})
    p = baselines.climatology(feats, clim)
    assert p[0] > 8 and p[1] < 1


def test_load_split_years() -> None:
    s = splits.load_split_years()
    assert s["test"] == [2024, 2025] and max(s["val"]) < min(s["test"])


def test_seasonal_ratio() -> None:
    idx = pd.date_range("2020-01-01", "2020-12-31")
    s = pd.Series(1.0, index=idx)
    assert hazards.seasonal_ratio(s, lpa_mm=122.0) == pytest.approx(1.0)  # Jun-Sep = 122 days


def test_spi_handles_all_zero_month_and_sparse_params() -> None:
    idx = pd.date_range("1981-01-01", periods=365 * 35, freq="D")
    rain = pd.Series(1.0, index=idx)
    rain[rain.index.month == 12] = 0.0  # December never rains
    t = spi.monthly_totals(rain, 1)
    p = spi.fit(t)
    z = spi.transform(t, p)
    assert np.isfinite(z.dropna()).all()
