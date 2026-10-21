"""Recompute EVERY number and figure used in the research paper from the repository. Nothing is typed by hand.

    python scripts/paper_numbers.py [OUT_DIR]        (default: ../paper/build)

Writes OUT_DIR/numbers.json, OUT_DIR/*.csv and figures (PNG). Takes several minutes: it retrains the models.
Confidence intervals: moving-block bootstrap over DAYS (block = 7 days, B = 1000). The 75 districts of one day share
the same weather, so the effective sample is the number of days, not the number of district-days.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import style
from src.features import build
from src.models import mos, train_eval
from src.whatif.engine import Scenario, WhatIfEngine

LAG_LEADS = (1, 2, 3, 5, 7)
MOS_LEADS = (1, 3, 7)
LAG_TREES = 250
MOS_TREES = 150
B = 1000
BLOCK = 7
HEAVY = 15.6
plt.rcParams.update(
    {"font.family": "STIXGeneral", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


# ----------------------------------------------------------------------------------------------- bootstrap
def block_idx(n_days: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n_days / BLOCK))
    starts = rng.integers(0, max(n_days - BLOCK + 1, 1), size=(B, n_blocks))
    return (starts[:, :, None] + np.arange(BLOCK)).reshape(B, -1)[:, :n_days]


def boot(
    comp: dict[str, np.ndarray], stat: Callable[[dict[str, np.ndarray]], np.ndarray], seed: int = 0
) -> dict[str, float]:
    n_days = len(next(iter(comp.values())))
    idx = block_idx(n_days, np.random.default_rng(seed))
    draws = stat({k: v[idx].sum(axis=1) for k, v in comp.items()})
    point = float(stat({k: np.array([v.sum()]) for k, v in comp.items()})[0])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"est": point, "lo": float(lo), "hi": float(hi)}


def per_day(df: pd.DataFrame, cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    g = pd.DataFrame(cols).groupby(pd.to_datetime(df["target_date"]).to_numpy()).sum().sort_index()
    return {c: g[c].to_numpy(dtype=float) for c in g.columns}


def skill_ci(df: pd.DataFrame, fcol: str) -> dict[str, float]:
    y = df["y"].to_numpy()
    comp = per_day(
        df, {"sf": (y - df[fcol].to_numpy()) ** 2, "sc": (y - df["climatology"].to_numpy()) ** 2, "n": np.ones(len(df))}
    )
    return boot(comp, lambda t: 1 - np.sqrt(t["sf"] / t["n"]) / np.sqrt(t["sc"] / t["n"]))


def bss_ci(df: pd.DataFrame, pcol: str) -> dict[str, float]:
    ev = (df["y"].to_numpy() >= 1.0).astype(float)
    comp = per_day(df, {"bf": (df[pcol].to_numpy() - ev) ** 2, "br": (df["p_ref"].to_numpy() - ev) ** 2})
    return boot(comp, lambda t: 1 - t["bf"] / t["br"])


def contingency(df: pd.DataFrame, fcol: str, thr: float) -> dict[str, np.ndarray]:
    o, f = df["y"].to_numpy() >= thr, df[fcol].to_numpy() >= thr
    return per_day(df, {"h": (o & f).astype(float), "m": (o & ~f).astype(float), "fa": (~o & f).astype(float)})


def heavy_scores(df: pd.DataFrame, fcol: str, thr: float = HEAVY) -> dict[str, float]:
    c = contingency(df, fcol, thr)
    h, m, fa = c["h"].sum(), c["m"].sum(), c["fa"].sum()
    out = {
        "obs_events": float(h + m),
        "pod": float(h / (h + m)) if h + m else float("nan"),
        "far": float(fa / (h + fa)) if h + fa else float("nan"),
        "csi": float(h / (h + m + fa)) if h + m + fa else float("nan"),
        "freq_bias": float((h + fa) / (h + m)) if h + m else float("nan"),
    }
    ci = boot(c, lambda t: t["h"] / (t["h"] + t["m"] + t["fa"]))
    out["csi_lo"], out["csi_hi"] = ci["lo"], ci["hi"]
    return out


def csi_diff_ci(df: pd.DataFrame, fa_col: str, fb_col: str, thr: float = HEAVY) -> dict[str, float]:
    a, b = contingency(df, fa_col, thr), contingency(df, fb_col, thr)
    comp = {f"a_{k}": v for k, v in a.items()} | {f"b_{k}": v for k, v in b.items()}
    return boot(
        comp, lambda t: t["a_h"] / (t["a_h"] + t["a_m"] + t["a_fa"]) - t["b_h"] / (t["b_h"] + t["b_m"] + t["b_fa"])
    )


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "paper" / "build"
    out.mkdir(parents=True, exist_ok=True)
    N: dict[str, object] = {}

    rain = pd.read_parquet(ROOT / "data/processed/district_rain_all.parquet")[["district_id", "date", "rain_mm"]]
    rain["date"] = pd.to_datetime(rain["date"])
    g = gpd.read_file(ROOT / "data/external/up_districts.geojson")
    rp = g.geometry.representative_point()
    cen = pd.DataFrame({"district_id": g["district_id"].astype(int), "lat": rp.y.to_numpy(), "lon": rp.x.to_numpy()})
    nwp = pd.read_parquet(ROOT / "data/processed/district_nwp.parquet")

    # ---- system / data facts
    area = g.to_crs("EPSG:6933").area / 1e6
    app = ROOT / "app_data"
    N["system"] = {
        "python": platform.python_version(),
        "versions": {
            p: md.version(p)
            for p in ("scikit-learn", "xgboost", "geopandas", "pytest", "streamlit", "plotly", "pandas", "numpy")
        },
        "n_districts": len(g),
        "district_area_km2_min": float(area.min()),
        "district_area_km2_median": float(area.median()),
        "district_area_km2_max": float(area.max()),
        "rain_rows": len(pd.read_parquet(app / "rain.parquet")),
        "temp_rows": len(pd.read_parquet(app / "temp.parquet")),
        "forecast_rows": len(pd.read_parquet(app / "forecast.parquet")),
        "app_data_mb": {f.name: round(f.stat().st_size / 1e6, 2) for f in sorted(app.glob("*.parquet"))},
        "app_data_total_mb": round(sum(f.stat().st_size for f in app.glob("*.parquet")) / 1e6, 2),
        "n_features_lag": len(build.FEATURE_COLS),
        "n_features_mos": len(mos.NWP_COLS),
        "years": [int(rain["date"].dt.year.min()), int(rain["date"].dt.year.max())],
        "nwp_rows": len(nwp),
    }
    col = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True
    )
    m = re.search(r"(\d+) tests? collected", col.stdout)
    N["system"]["n_tests"] = int(m.group(1)) if m else None  # type: ignore[index]

    # ---- lag-only models on the configured year-blocked splits
    cfg = train_eval.splits.load_split_years()
    plan = train_eval.plan_splits(sorted(rain["date"].dt.year.unique().tolist()), cfg["val"], cfg["test"])
    N["split"] = {
        "train": [plan.train[0], plan.train[-1]],
        "val": list(plan.val),
        "test": list(plan.test),
        "provisional": plan.provisional,
    }
    feats1 = build.build_features(rain, cen, leads=(1,))
    yr = feats1["target_date"].dt.year
    N["split"]["train_rows_per_lead"] = int((yr <= plan.train[-1]).sum())  # type: ignore[index]
    N["split"]["val_rows_per_lead"] = int(yr.isin(plan.val).sum())  # type: ignore[index]
    N["split"]["test_rows_per_lead"] = int(yr.isin(plan.test).sum())  # type: ignore[index]

    collected: list[pd.DataFrame] = []
    lag_tab = train_eval.evaluate(rain, cen, plan, LAG_LEADS, LAG_TREES, collect=collected)
    lag_tab.to_csv(out / "lag_metrics.csv", index=False)
    lag_pred = pd.concat(collected, ignore_index=True)
    lag_test = lag_pred[lag_pred["split"] == "test"]
    lag_ci: dict[str, dict[str, dict[str, float]]] = {}
    for k in LAG_LEADS:
        d = lag_test[lag_test["lead"] == k]
        lag_ci[str(k)] = {
            "skill": skill_ci(d, "xgb_hurdle"),
            "bss": bss_ci(d, "p_wet"),
            "persistence_skill": skill_ci(d, "persistence"),
        }
    N["lag"] = {"trees": LAG_TREES, "leads": list(LAG_LEADS), "ci": lag_ci}
    N["lag_table"] = json.loads(lag_tab.to_json(orient="records"))

    # ---- MOS (time-blocked: train 2024, test 2025)
    window = mos.MosWindow("2024-01-01", "2024-12-31", "2025-01-01")
    col2: list[pd.DataFrame] = []
    mos_tab = mos.evaluate(rain, nwp, cen, window, MOS_LEADS, MOS_TREES, collect=col2)
    mos_tab.to_csv(out / "mos_metrics.csv", index=False)
    mos_pred = pd.concat(col2, ignore_index=True)
    mos_ci: dict[str, dict[str, object]] = {}
    heavy: dict[str, dict[str, dict[str, float]]] = {}
    for k in MOS_LEADS:
        d = mos_pred[mos_pred["lead"] == k]
        mos_ci[str(k)] = {
            "skill": {f: skill_ci(d, f) for f in ("xgb_mos", "xgb_lag_full", "raw_nwp")},
            "bss": {"xgb_mos": bss_ci(d, "p_mos"), "xgb_lag_full": bss_ci(d, "p_lag_full")},
            "mos_minus_lag_skill": None,
        }
        heavy[str(k)] = {
            f: heavy_scores(d, f) for f in ("raw_nwp", "xgb_mos", "xgb_lag_full", "persistence", "climatology")
        }
        heavy[str(k)]["diff_nwp_minus_mos_csi"] = csi_diff_ci(d, "raw_nwp", "xgb_mos")
    N["mos"] = {
        "trees": MOS_TREES,
        "window": {"train": ["2024-01-01", "2024-12-31"], "test_from": "2025-01-01"},
        "test_rows_per_lead": int((mos_pred["lead"] == MOS_LEADS[0]).sum()),
        "ci": mos_ci,
        "heavy": heavy,
    }
    N["mos_table"] = json.loads(mos_tab.to_json(orient="records"))
    # observed frequency of heavy days in the test sets
    N["heavy_obs_rate"] = {
        "lag_test": float((lag_test[lag_test["lead"] == 1]["y"] >= HEAVY).mean()),
        "mos_test": float((mos_pred[mos_pred["lead"] == 1]["y"] >= HEAVY).mean()),
        "ge_64_5_lag_test": int((lag_test[lag_test["lead"] == 1]["y"] >= 64.5).sum()),
    }

    # ---- what-if on the deployed data and baseline
    r = pd.read_parquet(app / "rain.parquet")
    t = pd.read_parquet(app / "temp.parquet")
    s_rain = r.groupby("date")["rain_mm"].mean()
    s_rain.index = pd.DatetimeIndex(s_rain.index)
    s_tmax = t.groupby("date")["tmax"].mean()
    s_tmax.index = pd.DatetimeIndex(s_tmax.index)
    y0, y1 = int(s_rain.index.year.min()), int(s_rain.index.year.max())
    eng = WhatIfEngine(s_rain, s_tmax, (y0, y1), n_boot=B, seed=0)
    base = eng.run(Scenario())
    ratio = base.baseline["jjas_ratio_to_lpa"].sort_values()
    jm = s_rain[s_rain.index.month.isin([6, 7, 8, 9])]
    ann = s_rain.groupby(s_rain.index.year).sum()
    jj = jm.groupby(jm.index.year).sum()
    wi: dict[str, object] = {
        "baseline_years": [y0, y1],
        "n_years": len(eng.years),
        "lpa_mm": float(eng.lpa_mm),
        "n_boot": B,
        "jjas_share_of_annual_mean": float((jj / ann).mean()),
        "driest4": [int(v) for v in ratio.index[:4]],
        "driest4_ratio": [float(v) for v in ratio.values[:4]],
        "rank_2002": int(list(ratio.index).index(2002) + 1),
        "ratio_2002": float(ratio.loc[2002]),
        "drought_share_baseline": [base.drought_share_baseline, *base.drought_ci_baseline],
    }
    scen: dict[str, dict[str, object]] = {}
    for name, sc in {
        "rain_-20": Scenario(rain_factor=-0.20),
        "rain_-30": Scenario(rain_factor=-0.30),
        "extreme_+20": Scenario(extreme_factor=0.20),
        "temp_+1": Scenario(temp_delta=1.0),
        "temp_+2": Scenario(temp_delta=2.0),
        "temp_+3": Scenario(temp_delta=3.0),
        "rain_-20_temp_+2": Scenario(rain_factor=-0.20, temp_delta=2.0),
    }.items():
        res = eng.run(sc)
        scen[name] = {
            "drought_share": [res.drought_share_scenario, *res.drought_ci_scenario],
            "drought_delta_ci": list(res.drought_ci_delta),
            "summary": json.loads(res.summary.to_json(orient="index")),
        }
    wi["scenarios"] = scen
    wi["baseline_summary"] = json.loads(base.summary.to_json(orient="index"))
    N["whatif"] = wi

    # ---- figures
    fig_dir = out
    # Fig A: skill vs lead
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ks = list(LAG_LEADS)
    est = [lag_ci[str(k)]["skill"]["est"] * 100 for k in ks]
    lo = [est[i] - lag_ci[str(k)]["skill"]["lo"] * 100 for i, k in enumerate(ks)]
    hi = [lag_ci[str(k)]["skill"]["hi"] * 100 - est[i] for i, k in enumerate(ks)]
    ax[0].axhline(0, color="#444", lw=0.8)
    ax[0].errorbar(ks, est, yerr=[lo, hi], fmt="o-", color="#1F5F8B", capsize=3, label="History-only model (2024-25)")
    km = list(MOS_LEADS)
    e2 = [mos_ci[str(k)]["skill"]["xgb_mos"]["est"] * 100 for k in km]  # type: ignore[index]
    l2 = [e2[i] - mos_ci[str(k)]["skill"]["xgb_mos"]["lo"] * 100 for i, k in enumerate(km)]  # type: ignore[index]
    h2 = [mos_ci[str(k)]["skill"]["xgb_mos"]["hi"] * 100 - e2[i] for i, k in enumerate(km)]  # type: ignore[index]
    ax[0].errorbar(
        [k + 0.12 for k in km], e2, yerr=[l2, h2], fmt="s--", color="#B7791F", capsize=3, label="With NWP, MOS (2025)"
    )
    ax[0].set_xlabel("Lead time (days)")
    ax[0].set_ylabel("RMSE skill vs climatology (%)")
    ax[0].set_xticks(ks)
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    ax[0].set_title("(a) Error skill, 95% block-bootstrap interval", fontsize=8.5)
    b1 = [lag_ci[str(k)]["bss"]["est"] for k in ks]
    bl = [b1[i] - lag_ci[str(k)]["bss"]["lo"] for i, k in enumerate(ks)]
    bh = [lag_ci[str(k)]["bss"]["hi"] - b1[i] for i, k in enumerate(ks)]
    ax[1].axhline(0, color="#444", lw=0.8)
    ax[1].errorbar(ks, b1, yerr=[bl, bh], fmt="o-", color="#1F5F8B", capsize=3)
    ax[1].set_xlabel("Lead time (days)")
    ax[1].set_ylabel("Brier skill, rain >= 1 mm")
    ax[1].set_xticks(ks)
    ax[1].set_title("(b) Wet-day probability skill (seasonal reference)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_skill.png", dpi=220)
    plt.close(fig)

    # Fig B: heavy rain POD / FAR / frequency bias, lead 1 and 3
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    names = {"raw_nwp": "Raw NWP", "xgb_mos": "MOS", "xgb_lag_full": "History-only"}
    colors = {"raw_nwp": "#10345C", "xgb_mos": "#B7791F", "xgb_lag_full": "#8CC1C5"}
    for j, (metric, title) in enumerate(
        (("pod", "Hit rate (POD)"), ("far", "False-alarm ratio"), ("freq_bias", "Frequency bias (1 = unbiased)"))
    ):
        x = np.arange(2)
        for i, f in enumerate(names):
            vals = [heavy[str(k)][f][metric] for k in (1, 3)]  # type: ignore[index]
            ax[j].bar(x + (i - 1) * 0.26, vals, 0.26, color=colors[f], label=names[f])
        ax[j].set_xticks(x, ["1 day", "3 days"])
        ax[j].set_title(title, fontsize=8.5)
        if metric == "freq_bias":
            ax[j].axhline(1, color="#444", lw=0.8, ls=":")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Rain of at least {HEAVY} mm (district average), 2025 test", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_heavy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Fig C: what-if
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    yrs = base.baseline.index.to_numpy()
    sc20 = eng.run(Scenario(rain_factor=-0.20))
    ax[0].bar(yrs - 0.2, base.baseline["jjas_ratio_to_lpa"] * 100, 0.4, color="#BFDCDB", label="Observed")
    ax[0].bar(yrs + 0.2, sc20.perturbed["jjas_ratio_to_lpa"] * 100, 0.4, color="#2F7FA6", label="Rain -20%")
    ax[0].axhline(75, color="#B7791F", lw=1.6, label="75% of normal")
    ax[0].set_ylabel("Monsoon rain, % of mean")
    ax[0].set_xlabel("Year")
    ax[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax[0].set_title("", fontsize=8)
    dts = [0, 1, 2, 3]
    hw, lo_, hi_ = [], [], []
    for d in dts:
        res = eng.run(Scenario(temp_delta=float(d)))
        row = res.summary.loc["heatwave_days"]
        hw.append(row["scenario"])
        lo_.append(row["scenario"] + (row["ci_low"] - row["delta"]))
        hi_.append(row["scenario"] + (row["ci_high"] - row["delta"]))
    ax[1].errorbar(
        dts, hw, yerr=[np.array(hw) - np.array(lo_), np.array(hi_) - np.array(hw)], fmt="o-", color="#B83A2E", capsize=3
    )
    ax[1].set_xlabel("Warming applied to daily Tmax (C)")
    ax[1].set_ylabel("Heat-wave days per year")
    ax[1].set_xticks(dts)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_whatif.png", dpi=220)
    plt.close(fig)
    N["whatif"]["heatwave_vs_dT"] = {str(d): [float(h), float(a), float(b)] for d, h, a, b in zip(dts, hw, lo_, hi_)}  # type: ignore[index]

    # Fig D: map of the 2025 monsoon (what the dashboard shows)
    win = r[(r["date"].dt.year == 2025) & r["date"].dt.month.between(6, 9)].groupby("district_id")["rain_mm"].sum()
    gm = g.merge(win.rename("jjas"), left_on="district_id", right_index=True)
    cmap = LinearSegmentedColormap.from_list("monsoon", style.RAIN_SCALE)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    gm.plot(
        column="jjas",
        cmap=cmap,
        linewidth=0.3,
        edgecolor="white",
        legend=True,
        ax=ax,
        legend_kwds={"shrink": 0.7, "label": "June-Sept 2025 rain (mm)"},
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_map.png", dpi=220)
    plt.close(fig)
    N["map"] = {"jjas2025_mean_mm": float(win.mean()), "min": float(win.min()), "max": float(win.max())}

    (out / "numbers.json").write_text(json.dumps(N, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out / 'numbers.json'}")


if __name__ == "__main__":
    main()
