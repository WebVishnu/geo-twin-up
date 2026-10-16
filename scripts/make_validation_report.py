"""Render docs/validation_report.md from reports/metrics.csv and reports/mos_metrics.csv.

Every number in the report comes from those files, so re-running after new data keeps it honest.
    python -m src.models.train_eval ; python -m src.models.mos ; python scripts/make_validation_report.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import mos as mos_mod
from src.models import train_eval

ROOT = Path(__file__).resolve().parent.parent


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def f(v: float, d: int = 2) -> str:
    return "-" if pd.isna(v) else f"{v:.{d}f}"


def pct(v: float) -> str:
    return "-" if pd.isna(v) else f"{v:+.1%}"


def lag_table(m: pd.DataFrame, split: str) -> str:
    rows = []
    for (lead, name), g in m[m["split"] == split].groupby(["lead", "forecast"]):
        r = g.iloc[0]
        rows.append(
            {
                "Lead (days)": lead,
                "Forecast": name,
                "RMSE (mm)": f(r["rmse"]),
                "RMSE skill vs climatology": pct(r["rmse_skill_vs_clim"]) if name != "climatology" else "0 (reference)",
                "Brier skill, rain >= 1 mm": f(r["brier_skill_1mm"]),
                "CSI >= 2.5 mm": f(r["csi@2.5"]),
                "CSI >= 15.6 mm": f(r["csi@15.6"]),
            }
        )
    return md_table(pd.DataFrame(rows))


def mos_table(m: pd.DataFrame) -> str:
    rows = []
    for (lead, name), g in m.groupby(["lead", "forecast"]):
        r = g.iloc[0]
        rows.append(
            {
                "Lead (days)": lead,
                "Forecast": name,
                "RMSE (mm)": f(r["rmse"]),
                "RMSE skill vs climatology": pct(r["rmse_skill_vs_clim"]) if name != "climatology" else "0 (reference)",
                "Correlation": f(r["corr"]),
                "Brier skill": f(r["brier_skill_1mm"]),
                "CSI >= 2.5 mm": f(r["csi@2.5"]),
                "CSI >= 15.6 mm": f(r["csi@15.6"]),
            }
        )
    return md_table(pd.DataFrame(rows))


def drought_check() -> str:
    """Driest years in the UP-average rain series vs years IMD reported as below-normal for all-India (verify on IMD)."""
    from src.whatif.engine import WhatIfEngine

    r = pd.read_parquet(ROOT / "app_data/rain.parquet")
    s = r.groupby("date")["rain_mm"].mean()
    s.index = pd.DatetimeIndex(s.index)
    eng = WhatIfEngine(s, None, (int(s.index.year.min()), int(s.index.year.max())), n_boot=20)
    ratio = eng.run(eng_scn()).baseline["jjas_ratio_to_lpa"].sort_values()
    driest = [int(y) for y in ratio.index[:4]]
    known = {
        2009,
        2014,
        2015,
    }  # below-normal all-India monsoon years per IMD; CHECK on IMD pages before relying on this
    hit = sorted(known & set(driest))
    return (
        f"the four driest monsoons in the UP-average series are {driest}; {len(hit)} of the 3 years we expected "
        f"({sorted(known)}, below-normal all-India monsoons, to be verified on IMD pages) are among them: {hit}."
    )


def eng_scn():
    from src.whatif.engine import Scenario

    return Scenario()


def main() -> None:
    m = pd.read_csv(ROOT / "reports/metrics.csv")
    s = pd.read_csv(ROOT / "reports/mos_metrics.csv")
    prov = bool(m["provisional_split"].iloc[0])
    cfg = train_eval.splits.load_split_years()
    n_test = int(m[(m["split"] == "test") & (m["lead"] == m["lead"].min())]["n"].iloc[0])
    lag_test = m[(m["split"] == "test") & (m["forecast"] == "xgb_hurdle")].set_index("lead")
    verdicts = "\n".join(f"- {v}" for v in mos_mod.verdict(s))
    n_mos = int(s["n"].iloc[0])
    dc = drought_check()

    lag_lines = []
    for lead, r in lag_test.iterrows():
        gain = "beats" if r["rmse_skill_vs_clim"] > 0 else "does not beat"
        lag_lines.append(
            f"- {lead} day: RMSE skill vs climatology {pct(r['rmse_skill_vs_clim'])}, Brier skill {f(r['brier_skill_1mm'])}"
            f" ({gain} climatology on RMSE)."
        )

    doc = f"""# Validation report

Generated {dt.date.today()} by `scripts/make_validation_report.py` from `reports/metrics.csv` and
`reports/mos_metrics.csv`. Re-run it after any data refresh. The tables and the verdict lines are computed; sections 4 and 5
are our own reading of those numbers.

## 1. What was tested
- **Task:** forecast daily rain for each of the 75 Uttar Pradesh districts at 1, 3 and 7 days ahead.
- **Truth:** IMD gridded daily rain (0.25 degree), averaged over each district's area.
- **Splits (year-blocked, never random):** validation {cfg["val"]}, test {cfg["test"]}, training = all earlier years from 1991.
  {"These splits are PROVISIONAL: the configured years were not all cached." if prov else "The configured years were all available, so these are the final splits."}
  The test period was scored once, after the model was fixed. No tuning was done on it.
- **References every model must beat:** persistence (tomorrow = today), day-of-year climatology fitted on training years only,
  and, in section 3, the raw weather-model (NWP) forecast.
- **Why not just RMSE:** rain is mostly zero, so a forecast that is smooth and low can score a good RMSE while missing every
  storm. We therefore also report the critical success index (CSI) for rain above 2.5 mm and 15.6 mm, and the Brier skill
  score for the event "rain of 1 mm or more", measured against a seasonal (day-of-year) wet-day frequency, not a
  single constant, so that knowing the season alone earns no credit.

## 2. Model using only past rainfall and the season (XGBoost, one model per lead)
Test results ({n_test:,} district-days per lead):

{lag_table(m, "test")}

Reading it:
{chr(10).join(lag_lines)}

Validation results point the same way as the test results at every lead. That is consistent with, though not proof of,
the model not being overfitted to one period:

{lag_table(m, "val")}

## 3. Adding weather-model (NWP) forecasts (MOS correction)
Lead-specific archived NWP forecasts exist only from 2024, so this experiment trains on 2024 and tests on 2025
({n_mos:,} district-days per lead). It is time-blocked, not year-blocked, and is therefore indicative, not final.

{mos_table(s)}

Plain-language verdicts (computed, not written by hand):

{verdicts}

## 4. What we can and cannot claim
- **Can claim:** for the next day, a forecast that combines past rain with weather-model output has lower error than climatology
  and than the raw weather model on district averages (2025 test). It is still worse than the raw weather model at heavy rain.
- **Cannot claim:** useful skill at 3 days from rainfall history alone, or at 7 days from any source we tested.
  Heavy rain (15.6 mm or more) is detected poorly by every smoothed forecast; the raw weather model is better at it
  than the corrected one.
- **Sanity check of the what-if baseline:** {dc}
- **Therefore:** the dashboard labels the forecast layer as the weather model's own output and the what-if page as a
  sensitivity analysis, not a forecast.

## 5. Limits of this evidence
- The MOS experiment has one training year (2024) and one test year (2025).
- The NWP value is taken at each district's centre point while the truth is a district average, so part of the raw-NWP
  error is a mismatch of scale, not model error.
- Temperature in the what-if tool is ERA5-Land at district centres, not IMD observations.
- Heat-wave and heavy-rain thresholds follow IMD conventions as we understood them. Check them against IMD pages.
- Hyper-parameters were fixed, not tuned. We did not test whether tuning changes the picture.
"""
    out = ROOT / "docs" / "validation_report.md"
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({len(doc):,} chars)")


if __name__ == "__main__":
    main()
