import numpy as np
import pandas as pd

from src.indices import hazards, spi


def baseline_daily(years=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1981-01-01", periods=365 * years, freq="D")
    season = 1 + 3 * np.exp(-((idx.dayofyear - 200) ** 2) / (2 * 40**2))  # monsoon bump
    rain = rng.gamma(0.5, 6.0, len(idx)) * season * (rng.random(len(idx)) < 0.35)
    return pd.Series(rain, index=idx)


def test_baseline_spi_is_standardised():
    d = baseline_daily()
    t = spi.monthly_totals(d, 3)
    z = spi.transform(t, spi.fit(t))
    assert abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.1


def test_frozen_params_keep_the_perturbation_visible():
    d = baseline_daily()
    p = spi.fit(spi.monthly_totals(d, 3))
    z_dry = spi.transform(spi.monthly_totals(d * 0.7, 3), p)  # -30% rain, frozen baseline
    assert z_dry.mean() < -0.5
    p_bad = spi.fit(spi.monthly_totals(d * 0.7, 3))  # the bug: refit on perturbed
    assert abs(spi.transform(spi.monthly_totals(d * 0.7, 3), p_bad).mean()) < 0.1


def test_zero_perturbation_is_identity():
    d = baseline_daily()
    t = spi.monthly_totals(d, 3)
    p = spi.fit(t)
    assert np.allclose(spi.transform(t, p).dropna(), spi.transform(spi.monthly_totals(d * 1.0, 3), p).dropna())


def test_fit_rejects_short_baseline():
    d = baseline_daily(years=10)
    try:
        spi.fit(spi.monthly_totals(d, 3))
        assert False, "should have raised"
    except ValueError:
        pass


def test_imd_drought_classes():
    assert hazards.drought_class(0.80) == "normal"  # -20% rain: not drought
    assert hazards.drought_class(0.70) == "moderate drought"  # -30%: drought
    assert hazards.drought_class(0.40) == "severe drought"


def test_rx_indices():
    idx = pd.date_range("2020-01-01", "2021-12-31")
    s = pd.Series(0.0, index=idx)
    s["2020-07-10"] = 80
    s["2020-07-11"] = 20
    s["2021-08-01"] = 30
    assert hazards.rx1day(s)[2020] == 80 and hazards.rx5day(s)[2020] == 100
    assert hazards.heavy_rain_days(s)[2020] == 1 and hazards.heavy_rain_days(s)[2021] == 0


def test_heatwave_needs_two_days_and_uses_frozen_normal():
    idx = pd.date_range("2022-05-01", periods=6)
    normal = pd.Series(38.0, index=idx)
    tmax = pd.Series([38, 43, 43, 38, 43, 38.0], index=idx)  # one 2-day run, one isolated hot day
    assert hazards.heatwave_days(tmax, normal).tolist() == [False, True, True, False, False, False]
    warmer = hazards.heatwave_days(tmax + 2, normal).sum()  # +2 C vs SAME normal -> more hot days
    assert warmer >= hazards.heatwave_days(tmax, normal).sum()
