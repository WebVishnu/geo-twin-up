"""Feature engineering for district-level rain forecasting (ticket T7).

One row = (district, issue_date, lead). Issue date t is the END of day t: every feature uses
data from day t or earlier; the target is rain on day t+lead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = list(range(0, 14))          # rain on day t, t-1, ... t-13
ROLL = [3, 7, 14, 30]
DEFAULT_LEADS = (1, 2, 3, 5, 7)
FEATURE_COLS = (
    [f"rain_lag{l}" for l in LAGS]
    + [f"rain_sum{w}" for w in ROLL]
    + ["state_mean_lag0", "state_mean_sum7", "doy_sin", "doy_cos", "lat", "lon"]
)


def to_wide(df: pd.DataFrame, value: str = "rain_mm") -> pd.DataFrame:
    """Long [district_id, date, value] -> complete daily index x district_id (gaps become NaN)."""
    w = df.pivot(index="date", columns="district_id", values=value).sort_index()
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D"))


def build_features(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    leads: tuple[int, ...] = DEFAULT_LEADS,
    value: str = "rain_mm",
) -> pd.DataFrame:
    """centroids: columns [district_id, lat, lon]. Returns long frame with FEATURE_COLS, y, lead, issue_date, target_date."""
    P = to_wide(df, value)
    state = P.mean(axis=1)
    cen = centroids.set_index("district_id")
    doy = P.index.dayofyear.values
    base = {}
    for l in LAGS:
        base[f"rain_lag{l}"] = P.shift(l)
    for w in ROLL:
        base[f"rain_sum{w}"] = P.rolling(w, min_periods=w).sum()   # window ends at t (inclusive)
    frames = []
    for k in leads:
        y = P.shift(-k)
        parts = {}
        for name, mat in base.items():
            parts[name] = mat.stack(future_stack=True)
        parts["y"] = y.stack(future_stack=True)
        f = pd.DataFrame(parts)
        f.index.names = ["issue_date", "district_id"]
        f = f.reset_index()
        f["state_mean_lag0"] = f.issue_date.map(state)
        f["state_mean_sum7"] = f.issue_date.map(state.rolling(7, min_periods=7).sum())
        d = f.issue_date.dt.dayofyear.values
        f["doy_sin"] = np.sin(2 * np.pi * d / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * d / 365.25)
        f["lat"] = f.district_id.map(cen["lat"])
        f["lon"] = f.district_id.map(cen["lon"])
        f["lead"] = k
        f["target_date"] = f.issue_date + pd.to_timedelta(k, unit="D")
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)
