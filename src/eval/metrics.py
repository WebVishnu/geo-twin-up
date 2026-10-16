"""Forecast verification metrics (ticket T8). Rain needs threshold scores, not just RMSE."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAIN_THRESHOLDS_MM = (1.0, 2.5, 15.6, 64.5)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def corr(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float("nan") if y.std() == 0 or p.std() == 0 else float(np.corrcoef(y, p)[0, 1])


def categorical(y, p, thr: float) -> dict:
    """POD, FAR, CSI for the event 'rain >= thr'."""
    o, f = np.asarray(y) >= thr, np.asarray(p) >= thr
    hits, miss, fa = int((o & f).sum()), int((o & ~f).sum()), int((~o & f).sum())
    nan = float("nan")
    return {
        "pod": hits / (hits + miss) if hits + miss else nan,
        "far": fa / (hits + fa) if hits + fa else nan,
        "csi": hits / (hits + miss + fa) if hits + miss + fa else nan,
    }


def skill_score(err_model: float, err_ref: float) -> float:
    """1 = perfect, 0 = no better than reference, <0 = worse. Use with MSE/RMSE/Brier."""
    return 1.0 - err_model / err_ref if err_ref else float("nan")


def brier(y, prob, thr: float) -> float:
    return float(np.mean((np.asarray(prob) - (np.asarray(y) >= thr)) ** 2))


def table(y, forecasts: dict, thresholds=RAIN_THRESHOLDS_MM) -> pd.DataFrame:
    """One row per forecast name: RMSE, MAE, corr, CSI@thr..., plus RMSE skill vs climatology if given."""
    rows = []
    for name, p in forecasts.items():
        r = {"forecast": name, "rmse": rmse(y, p), "mae": mae(y, p), "corr": corr(y, p)}
        for t in thresholds:
            r[f"csi@{t}"] = categorical(y, p, t)["csi"]
        rows.append(r)
    df = pd.DataFrame(rows)
    if "climatology" in forecasts:
        ref = df.loc[df.forecast == "climatology", "rmse"].iloc[0]
        df["rmse_skill_vs_clim"] = df["rmse"].map(lambda e: skill_score(e, ref))
    return df
