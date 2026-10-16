"""Standardised Precipitation Index with FROZEN baseline parameters (ticket T12).

fit() learns gamma parameters per calendar month from the baseline period only.
transform() scores ANY series (including a what-if perturbed one) against those frozen
parameters. Refitting on the perturbed series would cancel the perturbation (classic bug).
Zeros are handled with the mixed distribution H(x) = q + (1 - q) * G(x).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_YEARS = 30


def monthly_totals(daily: pd.Series, scale: int = 3) -> pd.Series:
    """Daily mm (DatetimeIndex) -> rolling `scale`-month totals indexed by month-end."""
    m = daily.resample("MS").sum(min_count=1)
    return m.rolling(scale, min_periods=scale).sum()


def fit(totals: pd.Series, min_years: int = MIN_YEARS) -> dict[int, tuple[float, float, float]]:
    """Per calendar month: (q_zero, gamma_shape, gamma_scale)."""
    params = {}
    for month in range(1, 13):
        x = totals[totals.index.month == month].dropna()
        if len(x) < min_years:
            raise ValueError(f"month {month}: only {len(x)} baseline years, need >= {min_years}")
        pos = x[x > 0]
        q = 1 - len(pos) / len(x)
        if len(pos) < 5:
            params[month] = (q, np.nan, np.nan)
            continue
        shape, _, scale = stats.gamma.fit(pos, floc=0)
        params[month] = (float(q), float(shape), float(scale))
    return params


def transform(totals: pd.Series, params: dict[int, tuple[float, float, float]]) -> pd.Series:
    out = pd.Series(np.nan, index=totals.index)
    for month, (q, shape, scale) in params.items():
        mask = (totals.index.month == month) & totals.notna()
        if not mask.any() or np.isnan(shape):
            continue
        x = totals[mask].to_numpy()
        cdf = q + (1 - q) * stats.gamma.cdf(np.where(x > 0, x, 1.0), a=shape, scale=scale)
        cdf = np.where(x > 0, cdf, q / 2 + 1e-9)             # all-zero months: midpoint of the zero mass
        out[mask] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out
