"""Reference forecasts every model must beat (ticket T8)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence(feats: pd.DataFrame) -> np.ndarray:
    """Tomorrow looks like today: forecast for t+lead = rain on day t."""
    return feats["rain_lag0"].to_numpy()


def fit_climatology(wide: pd.DataFrame, train_years: list[int], window: int = 15) -> pd.DataFrame:
    """Day-of-year mean per district using ONLY train years, circularly smoothed. Index = doy 1..366."""
    tr = wide[wide.index.year.isin(train_years)]
    clim = tr.groupby(tr.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = pd.concat([clim.iloc[-window:], clim, clim.iloc[:window]])
    sm = ext.rolling(window, center=True, min_periods=1).mean().iloc[window:-window]
    sm.index = range(1, 367)
    return sm


def climatology(feats: pd.DataFrame, clim: pd.DataFrame) -> np.ndarray:
    doy = pd.to_datetime(feats["target_date"]).dt.dayofyear.to_numpy()
    cols = clim.columns.get_indexer(feats["district_id"])
    return clim.to_numpy()[doy - 1, cols]
