from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import mos


def synth(seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """4 districts, 2022-2025. NWP = truth + noise for target days from 2024 on (so NWP carries real signal)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2025-06-30")
    rain_rows, nwp_rows = [], []
    for d in range(4):
        wet = rng.random(len(dates)) < 0.3
        amt = np.where(wet, rng.gamma(1.2, 8.0, len(dates)), 0.0)
        rain_rows.append(pd.DataFrame({"district_id": d, "date": dates, "rain_mm": amt}))
        for lead in (1, 3):
            noisy = np.clip(amt + rng.normal(0, 3.0 * lead, len(dates)), 0, None)
            df = pd.DataFrame({"district_id": d, "valid_date": dates, "lead": lead, "nwp_precip": noisy})
            nwp_rows.append(df[df.valid_date >= "2024-01-01"])
    cen = pd.DataFrame({"district_id": range(4), "lat": 26 + np.arange(4) * 0.5, "lon": 80.0})
    return pd.concat(rain_rows, ignore_index=True), pd.concat(nwp_rows, ignore_index=True), cen


def test_window_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        mos.MosWindow("2024-01-01", "2024-12-31", "2024-12-31")


def test_masks_do_not_overlap() -> None:
    w = mos.MosWindow("2024-01-01", "2024-12-31", "2025-01-01")
    t = pd.Series(pd.date_range("2023-06-01", "2025-03-01"))
    m = w.masks(t)
    assert not (m["lag_full"] & m["test"]).any() and not (m["recent"] & m["test"]).any()
    assert (m["recent"] <= m["lag_full"]).all()


def test_attach_nwp_matches_on_target_date_and_lead() -> None:
    rain, nwp, cen = synth()
    from src.features import build

    f = mos.attach_nwp(build.build_features(rain, cen, leads=(1,)), nwp)
    row = f[(f.district_id == 0) & (f.lead == 1)].iloc[100]
    exp = nwp[(nwp.district_id == 0) & (nwp.lead == 1) & (nwp.valid_date == row.target_date)]
    if exp.empty:
        assert np.isnan(row.nwp_precip)
    else:
        assert row.nwp_precip == pytest.approx(float(exp.nwp_precip.iloc[0]))


def test_mos_beats_lag_only_when_nwp_has_signal() -> None:
    rain, nwp, cen = synth()
    w = mos.MosWindow("2024-01-01", "2024-12-31", "2025-01-01")
    tab = mos.evaluate(rain, nwp, cen, w, leads=(1, 3), n_estimators=60)
    assert {"persistence", "climatology", "raw_nwp", "xgb_lag_full", "xgb_lag_recent", "xgb_mos"} == set(tab.forecast)
    r = tab[tab.lead == 3].set_index("forecast")["rmse"]
    assert r["xgb_mos"] < r["xgb_lag_recent"]  # NWP carries signal by construction
    assert r["xgb_mos"] < r["climatology"]


def test_verdict_has_one_line_per_lead() -> None:
    rain, nwp, cen = synth()
    w = mos.MosWindow("2024-01-01", "2024-12-31", "2025-01-01")
    lines = mos.verdict(mos.evaluate(rain, nwp, cen, w, leads=(1, 3), n_estimators=30))
    assert len(lines) == 2 and all("MOS" in ln and "climatology" in ln for ln in lines)


def test_no_test_rows_is_an_error() -> None:
    rain, nwp, cen = synth()
    w = mos.MosWindow("2024-01-01", "2024-12-31", "2030-01-01")
    with pytest.raises(ValueError):
        mos.evaluate(rain, nwp, cen, w, leads=(1,), n_estimators=10)
