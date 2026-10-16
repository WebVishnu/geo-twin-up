"""Reference forecasts every model must beat (ticket T8)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def persistence(feats: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Tomorrow looks like today: forecast for t+lead = rain on day t."""
    return feats["rain_lag0"].to_numpy(dtype=np.float64)


def fit_climatology(wide: pd.DataFrame, train_years: list[int], window: int = 15) -> pd.DataFrame:
    """Day-of-year mean per district using ONLY train years, circularly smoothed. Index = doy 1..366."""
    idx = pd.DatetimeIndex(wide.index)
    tr = wide[idx.year.isin(train_years)]
    doy = pd.DatetimeIndex(tr.index).dayofyear
    clim = tr.groupby(doy).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = pd.concat([clim.iloc[-window:], clim, clim.iloc[:window]])
    sm = ext.rolling(window, center=True, min_periods=1).mean().iloc[window:-window]
    sm.index = pd.Index(range(1, 367))
    return sm


def climatology(feats: pd.DataFrame, clim: pd.DataFrame) -> npt.NDArray[np.float64]:
    doy = pd.to_datetime(feats["target_date"]).dt.dayofyear.to_numpy()
    cols = clim.columns.get_indexer(pd.Index(feats["district_id"]))
    return clim.to_numpy(dtype=np.float64)[doy - 1, cols]
