from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import train_eval as te


def synth(
    years: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020, 2021), n_dist: int = 4, seed: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.concat([pd.Series(pd.date_range(f"{y}-01-01", f"{y}-12-31")) for y in years], ignore_index=True)
    for d in range(n_dist):
        wet = rng.random(len(dates)) < 0.3
        # persistence in wetness gives a learnable signal
        for i in range(1, len(wet)):
            wet[i] = wet[i] or (wet[i - 1] and rng.random() < 0.5)
        rows.append(
            pd.DataFrame(
                {"district_id": d, "date": dates, "rain_mm": np.where(wet, rng.gamma(1.2, 8.0, len(dates)), 0.0)}
            )
        )
    cen = pd.DataFrame({"district_id": range(n_dist), "lat": 26 + np.arange(n_dist) * 0.5, "lon": 80.0})
    return pd.concat(rows, ignore_index=True), cen


def test_plan_uses_configured_years_when_present() -> None:
    p = te.plan_splits(list(range(1991, 2026)), [2022, 2023], [2024, 2025])
    assert not p.provisional and p.test == (2024, 2025) and max(p.train) == 2021


def test_plan_falls_back_and_says_so() -> None:
    p = te.plan_splits([1991, 1992, 1993, 1994, 1995, 1996, 2023], [2022, 2023], [2024, 2025])
    assert p.provisional and p.test == (1996, 2023) and p.val == (1994, 1995)
    assert "PROVISIONAL" in p.describe()
    assert set(p.train).isdisjoint(p.val + p.test)


def test_plan_needs_enough_years() -> None:
    with pytest.raises(ValueError):
        te.plan_splits([2020, 2021], [2022], [2023])


def test_evaluate_shapes_baselines_and_no_leakage() -> None:
    rain, cen = synth()
    plan = te.plan_splits(sorted(rain.date.dt.year.unique().tolist()), [2020], [2021])
    tab = te.evaluate(rain, cen, plan, leads=(1, 3), n_estimators=30)
    assert set(tab["split"]) == {"val", "test"} and set(tab["lead"]) == {1, 3}
    assert {"persistence", "climatology", "xgb_hurdle"} == set(tab["forecast"])
    assert {"rmse", "mae", "corr", "csi@2.5", "rmse_skill_vs_clim", "brier_skill_1mm", "n"} <= set(tab.columns)
    # climatology scored against itself has zero skill by definition
    clim = tab[tab["forecast"] == "climatology"]
    assert np.allclose(clim["rmse_skill_vs_clim"], 0.0)


def test_model_learns_wet_spell_persistence_at_lead_1() -> None:
    rain, cen = synth(years=(2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021))
    plan = te.plan_splits(sorted(rain.date.dt.year.unique().tolist()), [2020], [2021])
    tab = te.evaluate(rain, cen, plan, leads=(1,), n_estimators=60)
    t = tab[(tab["split"] == "test")].set_index("forecast")
    assert t.loc["xgb_hurdle", "rmse"] <= t.loc["climatology", "rmse"] + 1e-6
    assert t.loc["xgb_hurdle", "brier_skill_1mm"] > 0


def test_verdict_text_mentions_every_lead() -> None:
    rain, cen = synth()
    plan = te.plan_splits(sorted(rain.date.dt.year.unique().tolist()), [2020], [2021])
    lines = te.verdict(te.evaluate(rain, cen, plan, leads=(1, 3), n_estimators=20))
    assert len(lines) == 2 and all("RMSE" in ln for ln in lines)
