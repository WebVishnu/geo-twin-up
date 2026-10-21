"""Train and evaluate the rain models against honest baselines (ticket T9 / T8).

Run:  python -m src.models.train_eval            (uses every cached IMD year)

Rules this module enforces:
* splits are year-blocked and fixed by configs/base.yaml; if those years are not cached yet, it
  falls back to PROVISIONAL splits and says so in the output, never silently;
* climatology is fit on train years only;
* the test split is scored once, after the model is fixed (no tuning on it).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import baselines, metrics, splits
from src.features import build
from src.models import xgb_rain

WET_MM = 1.0


@dataclass(frozen=True)
class SplitPlan:
    """Which calendar years of TARGET dates belong to which split."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    provisional: bool

    def describe(self) -> str:
        tag = " (PROVISIONAL: configured years not cached yet)" if self.provisional else ""
        return f"train {list(self.train)} | val {list(self.val)} | test {list(self.test)}{tag}"


def plan_splits(available: list[int], val_years: list[int], test_years: list[int]) -> SplitPlan:
    """Use the configured holdout years if all are present; otherwise hold out the latest available years."""
    avail = sorted(set(available))
    if set(val_years + test_years) <= set(avail):
        train = tuple(y for y in avail if y not in val_years + test_years)
        return SplitPlan(train, tuple(val_years), tuple(test_years), provisional=False)
    if len(avail) < 5:
        raise ValueError(f"need at least 5 cached years to split; have {len(avail)}: {avail}")
    test, val = tuple(avail[-2:]), tuple(avail[-4:-2])
    train = tuple(avail[:-4])
    return SplitPlan(train, val, test, provisional=True)


def _split_masks(feats: pd.DataFrame, plan: SplitPlan) -> dict[str, np.ndarray]:
    yr = feats["target_date"].dt.year.to_numpy()
    return {
        name: np.isin(yr, list(years))
        for name, years in (("train", plan.train), ("val", plan.val), ("test", plan.test))
    }


def evaluate(
    rain: pd.DataFrame,
    centroids: pd.DataFrame,
    plan: SplitPlan,
    leads: tuple[int, ...] = (1, 3, 7),
    n_estimators: int = 200,
    seed: int = 42,
    collect: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fit per-lead hurdle models on train years and score val/test against persistence and climatology.

    Returns one row per (split, lead, forecast) with RMSE, MAE, correlation, CSI at the rain thresholds,
    RMSE skill vs climatology, and for the model a Brier skill score for the event rain >= 1 mm.
    If `collect` is a list, the per-row predictions of every split/lead are appended to it (for CIs and diagnostics).
    """
    feats = build.build_features(rain, centroids, leads=leads)
    masks = _split_masks(feats, plan)
    wide = build.to_wide(rain)
    clim = baselines.fit_climatology(wide, list(plan.train))
    wet_ind = wide.where(wide.isna(), (wide >= WET_MM).astype(float))
    p_wet_clim = baselines.fit_climatology(wet_ind, list(plan.train))
    models = xgb_rain.fit_per_lead(feats, masks["train"], n_estimators=n_estimators, seed=seed)

    rows: list[pd.DataFrame] = []
    for split in ("val", "test"):
        sub = feats[masks[split]]
        for k in leads:
            s = sub[sub["lead"] == k]
            if s.empty or k not in models:
                continue
            y = s["y"].to_numpy()
            pred = xgb_rain.predict_per_lead({k: models[k]}, s)
            tab = metrics.table(
                y,
                {
                    "persistence": baselines.persistence(s),
                    "climatology": baselines.climatology(s, clim),
                    "xgb_hurdle": pred["rain_mm_pred"].to_numpy(),
                },
            )
            # Reference probability: seasonal (day-of-year, per district) wet-day frequency from TRAIN years.
            # A single all-year constant would let the model "win" just by knowing the season.
            p_ref = baselines.climatology(s, p_wet_clim)
            bs_model = metrics.brier(y, pred["p_wet"].to_numpy(), WET_MM)
            bs_clim = metrics.brier(y, p_ref, WET_MM)
            tab["brier_skill_1mm"] = np.nan
            tab.loc[tab["forecast"] == "xgb_hurdle", "brier_skill_1mm"] = metrics.skill_score(bs_model, bs_clim)
            tab.insert(0, "lead", k)
            tab.insert(0, "split", split)
            tab["n"] = len(y)
            rows.append(tab)
            if collect is not None:
                collect.append(
                    pd.DataFrame(
                        {
                            "split": split,
                            "lead": k,
                            "district_id": s["district_id"].to_numpy(),
                            "target_date": s["target_date"].to_numpy(),
                            "y": y,
                            "persistence": baselines.persistence(s),
                            "climatology": baselines.climatology(s, clim),
                            "xgb_hurdle": pred["rain_mm_pred"].to_numpy(),
                            "p_wet": pred["p_wet"].to_numpy(),
                            "p_ref": p_ref,
                        }
                    )
                )
    if not rows:
        raise ValueError("no rows to evaluate: check that val/test years contain data")
    return pd.concat(rows, ignore_index=True)


def verdict(table: pd.DataFrame) -> list[str]:
    """Plain-language lines: does the model beat climatology and persistence on TEST?"""
    out = []
    t = table[table["split"] == "test"]
    for k, g in t.groupby("lead"):
        r = g.set_index("forecast")["rmse"]
        csi = g.set_index("forecast")["csi@2.5"]
        out.append(
            f"lead {k}d: RMSE xgb {r['xgb_hurdle']:.2f} vs clim {r['climatology']:.2f} vs persist {r['persistence']:.2f}"
            f" | CSI@2.5mm xgb {csi['xgb_hurdle']:.2f} vs persist {csi['persistence']:.2f}"
            f" -> {'beats' if r['xgb_hurdle'] < r['climatology'] else 'does NOT beat'} climatology on RMSE"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rain", default="data/processed/district_rain_all.parquet")
    ap.add_argument("--districts", default="data/external/up_districts.geojson")
    ap.add_argument("--out", default="reports/metrics.csv")
    ap.add_argument("--leads", default="1,3,7")
    ap.add_argument("--trees", type=int, default=200)
    a = ap.parse_args()

    import geopandas as gpd  # local import: not needed by the library functions

    rain = pd.read_parquet(a.rain)[["district_id", "date", "rain_mm"]]
    rain["date"] = pd.to_datetime(rain["date"])
    g = gpd.read_file(a.districts)
    rp = g.geometry.representative_point()
    cen = pd.DataFrame({"district_id": g["district_id"].astype(int), "lat": rp.y.to_numpy(), "lon": rp.x.to_numpy()})

    cfg = splits.load_split_years()
    plan = plan_splits(sorted(rain["date"].dt.year.unique().tolist()), cfg["val"], cfg["test"])
    print("splits:", plan.describe(), flush=True)
    tab = evaluate(rain, cen, plan, tuple(int(x) for x in a.leads.split(",")), a.trees)
    tab["provisional_split"] = plan.provisional
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out, index=False)
    print("\n".join(verdict(tab)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
