"""What-if sensitivity engine (ticket T13).

Perturbs OBSERVED history with the delta method (rain: multiplicative, temperature: additive)
and recomputes hazard indicators against FROZEN baseline parameters. It never calls the ML
models: a world with 30% less rain lies outside what they learned, so their output would be
extrapolation. Results are sensitivity analysis, not forecasts, and are always labelled so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.indices import hazards, spi

LABEL = "Sensitivity analysis, not a forecast"
MONSOON = (6, 7, 8, 9)


def _dt(s: pd.Series) -> pd.DatetimeIndex:
    """Series index as a DatetimeIndex (satisfies the type checker; raises if not datetime-like)."""
    return pd.DatetimeIndex(s.index)


@dataclass(frozen=True)
class Scenario:
    """A hypothetical change applied to the observed baseline.

    Attributes:
        rain_factor: Fractional change in all rainfall, e.g. -0.2 for 20% less rain.
        extreme_factor: Extra fractional change on heavy-rain days only (days at or above the
            baseline 95th percentile of wet days). Lets users test intensification separately.
        temp_delta: Additive change in daily maximum temperature, in degrees C.
    """

    rain_factor: float = 0.0
    extreme_factor: float = 0.0
    temp_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.rain_factor <= -1.0 or self.extreme_factor <= -1.0:
            raise ValueError("factors must be > -1 (cannot remove more than 100% of rain)")
        if not -10.0 <= self.temp_delta <= 10.0:
            raise ValueError("temp_delta outside the supported range of -10..+10 C")

    @property
    def is_null(self) -> bool:
        return self.rain_factor == 0.0 and self.extreme_factor == 0.0 and self.temp_delta == 0.0


@dataclass
class WhatIfResult:
    """Baseline-vs-scenario indicators for one district (or area average)."""

    label: str
    scenario: Scenario
    baseline: pd.DataFrame  # one row per year
    perturbed: pd.DataFrame  # same columns
    summary: pd.DataFrame  # per indicator: baseline, scenario, delta, ci_low, ci_high
    drought_share_baseline: float = field(default=0.0)
    drought_share_scenario: float = field(default=0.0)


class WhatIfEngine:
    """Holds frozen baseline statistics and evaluates scenarios against them.

    Args:
        rain: Daily rainfall in mm with a DatetimeIndex.
        tmax: Optional daily maximum temperature in C on the same index.
        baseline_years: Inclusive (start, end) years used to freeze all reference statistics.
        n_boot: Bootstrap resamples over years for the uncertainty band.
        seed: Seed for the bootstrap, so results are reproducible.
    """

    def __init__(
        self,
        rain: pd.Series,
        tmax: pd.Series | None = None,
        baseline_years: tuple[int, int] = (1991, 2020),
        n_boot: int = 200,
        seed: int = 0,
    ) -> None:
        y0, y1 = baseline_years
        self.rain = rain.sort_index()
        self.tmax = tmax.sort_index() if tmax is not None else None
        self.years = [y for y in sorted(set(_dt(self.rain).year)) if y0 <= y <= y1]
        if len(self.years) < spi.MIN_YEARS:
            raise ValueError(f"baseline has {len(self.years)} years; SPI needs >= {spi.MIN_YEARS}")
        self._base_rain = self.rain[_dt(self.rain).year.isin(self.years)]
        self.n_boot, self.seed = n_boot, seed

        self._spi_params = spi.fit(spi.monthly_totals(self._base_rain, 3))
        jjas = self._base_rain[_dt(self._base_rain).month.isin(MONSOON)]
        self.lpa_mm = float(jjas.groupby(_dt(jjas).year).sum().mean())
        wet = self._base_rain[self._base_rain > 1.0]
        self.extreme_threshold = float(wet.quantile(0.95)) if len(wet) else np.inf
        self._tmax_normal: pd.Series | None = None
        if self.tmax is not None:
            base_t = self.tmax[_dt(self.tmax).year.isin(self.years)]
            self._tmax_normal = base_t.groupby(_dt(base_t).dayofyear).mean()

    # ------------------------------------------------------------------ perturbation
    def perturb(self, scn: Scenario) -> tuple[pd.Series, pd.Series | None]:
        """Apply the delta method to the observed series. Never refits any parameter."""
        rain = self.rain * (1.0 + scn.rain_factor)
        if scn.extreme_factor:
            heavy = self.rain >= self.extreme_threshold
            rain = rain.where(~heavy, rain * (1.0 + scn.extreme_factor))
        tmax = None if self.tmax is None else self.tmax + scn.temp_delta
        return rain, tmax

    # ------------------------------------------------------------------ indicators
    def _indicators(self, rain: pd.Series, tmax: pd.Series | None) -> pd.DataFrame:
        yrs = [y for y in self.years if y in set(_dt(rain).year)]
        r = rain[_dt(rain).year.isin(yrs)]
        monsoon_r = r[_dt(r).month.isin(MONSOON)]
        jjas_total = monsoon_r.groupby(_dt(monsoon_r).year).sum()
        spi_series = spi.transform(spi.monthly_totals(r, 3), self._spi_params)  # FROZEN baseline params
        spi_jjas = spi_series[_dt(spi_series).month.isin(MONSOON)]
        out = pd.DataFrame(
            {
                "jjas_ratio_to_lpa": jjas_total / self.lpa_mm,
                "spi3_monsoon_mean": spi_jjas.groupby(_dt(spi_jjas).year).mean(),
                "rx1day_mm": hazards.rx1day(r),
                "rx5day_mm": hazards.rx5day(r),
                "heavy_rain_days": hazards.heavy_rain_days(r),
            }
        )
        out["drought"] = out["jjas_ratio_to_lpa"].map(lambda x: hazards.drought_class(float(x)) != "normal")
        if tmax is not None and self._tmax_normal is not None:
            t = tmax[_dt(tmax).year.isin(yrs)]
            normal = pd.Series(self._tmax_normal.reindex(_dt(t).dayofyear).to_numpy(), index=t.index)
            hw = hazards.heatwave_days(t, normal)  # departure vs FROZEN normal
            out["heatwave_days"] = hw.groupby(_dt(hw).year).sum()
        return out

    def _boot_ci(self, delta: np.ndarray) -> tuple[float, float]:
        rng = np.random.default_rng(self.seed)
        n = len(delta)
        means = rng.choice(delta, size=(self.n_boot, n), replace=True).mean(axis=1)
        lo, hi = np.percentile(means, [5, 95])
        return float(lo), float(hi)

    # ------------------------------------------------------------------ public API
    def run(self, scn: Scenario) -> WhatIfResult:
        """Evaluate a scenario and return baseline, perturbed and summary tables."""
        base = self._indicators(self._base_rain, None if self.tmax is None else self.tmax)
        rain_p, tmax_p = self.perturb(scn)
        pert = self._indicators(rain_p, tmax_p)
        rows = []
        for col in [c for c in base.columns if c != "drought"]:
            d = (pert[col] - base[col]).dropna().to_numpy(dtype=float)
            lo, hi = self._boot_ci(d) if len(d) else (np.nan, np.nan)
            rows.append(
                {
                    "indicator": col,
                    "baseline": float(base[col].mean()),
                    "scenario": float(pert[col].mean()),
                    "delta": float(d.mean()) if len(d) else np.nan,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
        return WhatIfResult(
            label=LABEL,
            scenario=scn,
            baseline=base,
            perturbed=pert,
            summary=pd.DataFrame(rows).set_index("indicator"),
            drought_share_baseline=float(base["drought"].mean()),
            drought_share_scenario=float(pert["drought"].mean()),
        )
