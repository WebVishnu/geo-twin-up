"""Simple, defined hazard indicators (ticket T12).

Thresholds follow IMD conventions as reported in the research notes. VERIFY against the official
IMD pages before presenting: heat-wave criteria and the 64.5 mm heavy-rain class in particular.
"""
from __future__ import annotations

import pandas as pd

HEAVY_RAIN_MM = 64.5


def drought_class(rain_ratio_to_lpa: float) -> str:
    """IMD seasonal meteorological drought: <75% of long-period average = drought."""
    if rain_ratio_to_lpa >= 0.75:
        return "normal"
    return "severe drought" if rain_ratio_to_lpa < 0.5 else "moderate drought"


def seasonal_ratio(daily: pd.Series, lpa_mm: float, months=(6, 7, 8, 9)) -> float:
    """Monsoon (JJAS) total as a fraction of the long-period average."""
    s = daily[daily.index.month.isin(months)].sum()
    return float(s / lpa_mm)


def rx1day(daily: pd.Series) -> pd.Series:
    return daily.groupby(daily.index.year).max()


def rx5day(daily: pd.Series) -> pd.Series:
    return daily.rolling(5, min_periods=5).sum().groupby(daily.index.year).max()


def heavy_rain_days(daily: pd.Series, thr: float = HEAVY_RAIN_MM) -> pd.Series:
    return (daily >= thr).groupby(daily.index.year).sum()


def heatwave_days(tmax: pd.Series, normal_tmax: pd.Series) -> pd.Series:
    """Plains criterion (verify!): Tmax >= 40 C AND departure from normal >= 4.5 C, OR Tmax >= 45 C,
    on at least 2 consecutive days. `normal_tmax` must be the FROZEN baseline normal (same index),
    otherwise a +2 C what-if shift cancels itself. Returns a boolean series of heat-wave days."""
    dep = tmax - normal_tmax
    hot = ((tmax >= 40) & (dep >= 4.5)) | (tmax >= 45)
    run = hot & (hot.shift(1, fill_value=False) | hot.shift(-1, fill_value=False))
    return run
