"""MOS: does adding NWP forecasts to the rain model help? (ticket T10)

Lead-specific NWP archives only exist from 2024, so everything here is TIME-blocked, not year-blocked:
train on targets up to `train_end`, test on targets from `test_start`. Nothing is tuned on the test period.

Six forecasts are scored on the same test rows:
  persistence, climatology, raw_nwp, xgb_lag_full (lags, all history), xgb_lag_recent (lags, same short
  window as MOS) and xgb_mos (lags + NWP, short window). The honest question is xgb_mos vs raw_nwp and
  vs xgb_lag_recent: it isolates what NWP adds when the training window is the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval import baselines, metrics
from src.features import build
from src.models import xgb_rain

WET_MM = 1.0
NWP_COLS = [*build.FEATURE_COLS, "nwp_precip"]


@dataclass(frozen=True)
class MosWindow:
    """Dates are TARGET dates. Training must end before testing starts (checked)."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str | None = None

    def __post_init__(self) -> None:
        if pd.Timestamp(self.train_end) >= pd.Timestamp(self.test_start):
            raise ValueError("train_end must be before test_start (no overlap allowed)")

    def masks(self, target: pd.Series) -> dict[str, np.ndarray]:
        t = pd.to_datetime(target)
        test = t >= pd.Timestamp(self.test_start)
        if self.test_end:
            test &= t <= pd.Timestamp(self.test_end)
        return {
            "lag_full": (t <= pd.Timestamp(self.train_end)).to_numpy(),
            "recent": ((t >= pd.Timestamp(self.train_start)) & (t <= pd.Timestamp(self.train_end))).to_numpy(),
            "test": test.to_numpy(),
        }


def attach_nwp(feats: pd.DataFrame, nwp: pd.DataFrame) -> pd.DataFrame:
    """Join the NWP forecast for each row's (district, target date, lead). Rows without NWP keep NaN."""
    n = nwp.rename(columns={"valid_date": "target_date"})
    n = n.assign(target_date=pd.to_datetime(n["target_date"]))
    return feats.merge(
        n[["district_id", "target_date", "lead", "nwp_precip"]], on=["district_id", "target_date", "lead"], how="left"
    )


def evaluate(
    rain: pd.DataFrame,
    nwp: pd.DataFrame,
    centroids: pd.DataFrame,
    window: MosWindow,
    leads: tuple[int, ...] = (1, 3, 7),
    n_estimators: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Score all six forecasts on the test window; one row per (lead, forecast)."""
    feats = attach_nwp(build.build_features(rain, centroids, leads=leads), nwp)
    m = window.masks(feats["target_date"])
    has_nwp = feats["nwp_precip"].notna().to_numpy()

    wide = build.to_wide(rain)
    train_years = sorted(
        {int(y) for y in pd.DatetimeIndex(wide.index).year if pd.Timestamp(y, 12, 31) < pd.Timestamp(window.test_start)}
    )
    clim = baselines.fit_climatology(wide, train_years)
    wet = wide.where(wide.isna(), (wide >= WET_MM).astype(float))
    p_clim = baselines.fit_climatology(wet, train_years)

    lag_full = xgb_rain.fit_per_lead(feats, m["lag_full"], n_estimators=n_estimators, seed=seed)
    lag_recent = xgb_rain.fit_per_lead(feats, m["recent"] & has_nwp, n_estimators=n_estimators, seed=seed)
    mos = xgb_rain.fit_per_lead(feats, m["recent"] & has_nwp, cols=NWP_COLS, n_estimators=n_estimators, seed=seed)

    test = feats[m["test"] & has_nwp]
    rows: list[pd.DataFrame] = []
    for k in leads:
        s = test[test["lead"] == k]
        if s.empty or k not in mos:
            continue
        y = s["y"].to_numpy()
        pl = xgb_rain.predict_per_lead({k: lag_full[k]}, s)
        pr = xgb_rain.predict_per_lead({k: lag_recent[k]}, s)
        pm = xgb_rain.predict_per_lead({k: mos[k]}, s, cols=NWP_COLS)
        tab = metrics.table(
            y,
            {
                "persistence": baselines.persistence(s),
                "climatology": baselines.climatology(s, clim),
                "raw_nwp": s["nwp_precip"].to_numpy(),
                "xgb_lag_full": pl["rain_mm_pred"].to_numpy(),
                "xgb_lag_recent": pr["rain_mm_pred"].to_numpy(),
                "xgb_mos": pm["rain_mm_pred"].to_numpy(),
            },
        )
        p_ref = baselines.climatology(s, p_clim)
        bs_ref = metrics.brier(y, p_ref, WET_MM)
        tab["brier_skill_1mm"] = np.nan
        for name, p in (("xgb_lag_full", pl), ("xgb_lag_recent", pr), ("xgb_mos", pm)):
            tab.loc[tab["forecast"] == name, "brier_skill_1mm"] = metrics.skill_score(
                metrics.brier(y, p["p_wet"].to_numpy(), WET_MM), bs_ref
            )
        tab.insert(0, "lead", k)
        tab["n"] = len(y)
        rows.append(tab)
    if not rows:
        raise ValueError("no test rows with NWP available: check the NWP download and the test window")
    return pd.concat(rows, ignore_index=True)


def verdict(table: pd.DataFrame) -> list[str]:
    """One plain line per lead: does MOS beat raw NWP and the lag-only model trained on the same window?"""
    out = []
    for k, g in table.groupby("lead"):
        r = g.set_index("forecast")["rmse"]
        c = g.set_index("forecast")["csi@2.5"]
        beats_nwp = r["xgb_mos"] < r["raw_nwp"]
        beats_lag = r["xgb_mos"] < r["xgb_lag_recent"]
        out.append(
            f"lead {k}d: RMSE mos {r['xgb_mos']:.2f} | raw NWP {r['raw_nwp']:.2f} | lag-only(recent) {r['xgb_lag_recent']:.2f}"
            f" | clim {r['climatology']:.2f} || CSI@2.5 mos {c['xgb_mos']:.2f} vs NWP {c['raw_nwp']:.2f}"
            f" -> MOS {'beats' if beats_nwp else 'does NOT beat'} raw NWP; {'beats' if beats_lag else 'does NOT beat'} lag-only"
        )
    return out


def main() -> None:
    """python -m src.models.mos  (needs IMD rain through the test period and data/processed/district_nwp.parquet)."""
    import argparse
    from pathlib import Path

    import geopandas as gpd

    ap = argparse.ArgumentParser()
    ap.add_argument("--rain", default="data/processed/district_rain_all.parquet")
    ap.add_argument("--nwp", default="data/processed/district_nwp.parquet")
    ap.add_argument("--districts", default="data/external/up_districts.geojson")
    ap.add_argument("--train-start", default="2024-01-01")
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--out", default="reports/mos_metrics.csv")
    a = ap.parse_args()

    rain = pd.read_parquet(a.rain)[["district_id", "date", "rain_mm"]]
    rain["date"] = pd.to_datetime(rain["date"])
    nwp = pd.read_parquet(a.nwp)
    g = gpd.read_file(a.districts)
    rp = g.geometry.representative_point()
    cen = pd.DataFrame({"district_id": g["district_id"].astype(int), "lat": rp.y.to_numpy(), "lon": rp.x.to_numpy()})
    w = MosWindow(a.train_start, a.train_end, a.test_start)
    print(f"train targets {w.train_start}..{w.train_end} | test from {w.test_start}", flush=True)
    tab = evaluate(rain, nwp, cen, w)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out, index=False)
    print("\n".join(verdict(tab)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
