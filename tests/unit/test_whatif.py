from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.whatif.engine import LABEL, Scenario, WhatIfEngine


def make_series(years: int = 40, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1981-01-01", periods=365 * years, freq="D")
    season = 1 + 4 * np.exp(-((idx.dayofyear - 200) ** 2) / (2 * 35**2))
    rain = rng.gamma(0.5, 7.0, len(idx)) * season * (rng.random(len(idx)) < 0.35)
    doy = idx.dayofyear
    tmax = 28 + 10 * np.sin(2 * np.pi * (doy - 90) / 365) + rng.normal(0, 2.0, len(idx))
    return pd.Series(rain, index=idx), pd.Series(tmax, index=idx)


@pytest.fixture(scope="module")
def engine() -> WhatIfEngine:
    rain, tmax = make_series()
    return WhatIfEngine(rain, tmax, baseline_years=(1981, 2020), n_boot=100)


def test_zero_perturbation_is_identity(engine: WhatIfEngine) -> None:
    r = engine.run(Scenario())
    assert (r.summary["delta"].abs() < 1e-9).all()
    assert r.drought_share_baseline == r.drought_share_scenario


def test_minus_30_percent_crosses_imd_drought_line(engine: WhatIfEngine) -> None:
    r30 = engine.run(Scenario(rain_factor=-0.30))
    r20 = engine.run(Scenario(rain_factor=-0.20))
    ratio = r30.summary.loc["jjas_ratio_to_lpa"]
    assert ratio["scenario"] == pytest.approx(0.7 * ratio["baseline"], rel=1e-6)  # delta method is exact
    assert r30.drought_share_scenario > r20.drought_share_scenario > r20.drought_share_baseline - 1e-9
    assert r30.drought_share_scenario > 0.5  # a -30% season is mostly below 75% of LPA


def test_frozen_params_keep_spi_shift_visible(engine: WhatIfEngine) -> None:
    r = engine.run(Scenario(rain_factor=-0.30))
    assert r.summary.loc["spi3_monsoon_mean", "delta"] < -0.4


def test_heatwave_days_monotonic_in_temperature(engine: WhatIfEngine) -> None:
    days = [engine.run(Scenario(temp_delta=t)).summary.loc["heatwave_days", "scenario"] for t in (0, 1, 2, 3)]
    assert days == sorted(days) and days[-1] > days[0]


def test_extreme_factor_raises_extremes_not_totals_much(engine: WhatIfEngine) -> None:
    r = engine.run(Scenario(extreme_factor=0.2))
    assert r.summary.loc["rx1day_mm", "delta"] > 0
    assert r.summary.loc["heavy_rain_days", "delta"] >= 0


def test_uncertainty_band_brackets_the_mean_and_is_reproducible(engine: WhatIfEngine) -> None:
    a = engine.run(Scenario(rain_factor=-0.2)).summary
    b = engine.run(Scenario(rain_factor=-0.2)).summary
    pd.testing.assert_frame_equal(a, b)
    assert (a["ci_low"] <= a["delta"] + 1e-9).all() and (a["delta"] <= a["ci_high"] + 1e-9).all()


def test_result_is_always_labelled(engine: WhatIfEngine) -> None:
    assert engine.run(Scenario(rain_factor=-0.1)).label == LABEL


@pytest.mark.parametrize("kwargs", [{"rain_factor": -1.0}, {"extreme_factor": -1.5}, {"temp_delta": 25.0}])
def test_invalid_scenarios_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Scenario(**kwargs)


def test_short_baseline_rejected() -> None:
    rain, tmax = make_series(years=10)
    with pytest.raises(ValueError):
        WhatIfEngine(rain, tmax, baseline_years=(1981, 1990))


def test_engine_never_imports_ml_models() -> None:
    tree = ast.parse(Path("src/whatif/engine.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("src.models", "xgboost", "tensorflow", "torch")) for m in imported)


def test_drought_share_band_brackets_the_estimate_and_is_reproducible(engine: WhatIfEngine) -> None:
    a = engine.run(Scenario(rain_factor=-0.2))
    b = engine.run(Scenario(rain_factor=-0.2))
    assert a.drought_ci_scenario == b.drought_ci_scenario
    lo, hi = a.drought_ci_scenario
    assert lo <= a.drought_share_scenario <= hi
    dlo, dhi = a.drought_ci_delta
    assert dlo <= a.drought_share_scenario - a.drought_share_baseline <= dhi
    # zero perturbation: the delta band must be exactly zero
    z = engine.run(Scenario())
    assert z.drought_ci_delta == (0.0, 0.0)
