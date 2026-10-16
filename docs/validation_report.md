# Validation report

Generated 2026-10-17 by `scripts/make_validation_report.py` from `reports/metrics.csv` and
`reports/mos_metrics.csv`. Re-run it after any data refresh. The tables and the verdict lines are computed; sections 4 and 5
are our own reading of those numbers.

## 1. What was tested
- **Task:** forecast daily rain for each of the 75 Uttar Pradesh districts at 1, 3 and 7 days ahead.
- **Truth:** IMD gridded daily rain (0.25 degree), averaged over each district's area.
- **Splits (year-blocked, never random):** validation [2022, 2023], test [2024, 2025], training = all earlier years from 1991.
  The configured years were all available, so these are the final splits.
  The test period was scored once, after the model was fixed. No tuning was done on it.
- **References every model must beat:** persistence (tomorrow = today), day-of-year climatology fitted on training years only,
  and, in section 3, the raw weather-model (NWP) forecast.
- **Why not just RMSE:** rain is mostly zero, so a forecast that is smooth and low can score a good RMSE while missing every
  storm. We therefore also report the critical success index (CSI) for rain above 2.5 mm and 15.6 mm, and the Brier skill
  score for the event "rain of 1 mm or more", measured against a seasonal (day-of-year) wet-day frequency, not a
  single constant, so that knowing the season alone earns no credit.

## 2. Model using only past rainfall and the season (XGBoost, one model per lead)
Test results (54,825 district-days per lead):

| Lead (days) | Forecast | RMSE (mm) | RMSE skill vs climatology | Brier skill, rain >= 1 mm | CSI >= 2.5 mm | CSI >= 15.6 mm |
|---|---|---|---|---|---|---|
| 1 | climatology | 7.24 | 0 (reference) | - | 0.42 | 0.00 |
| 1 | persistence | 8.51 | -17.5% | - | 0.39 | 0.17 |
| 1 | xgb_hurdle | 6.84 | +5.5% | 0.15 | 0.47 | 0.07 |
| 3 | climatology | 7.24 | 0 (reference) | - | 0.42 | 0.00 |
| 3 | persistence | 10.04 | -38.6% | - | 0.28 | 0.08 |
| 3 | xgb_hurdle | 7.36 | -1.6% | 0.01 | 0.43 | 0.00 |
| 7 | climatology | 7.24 | 0 (reference) | - | 0.42 | 0.00 |
| 7 | persistence | 10.02 | -38.3% | - | 0.26 | 0.08 |
| 7 | xgb_hurdle | 7.40 | -2.1% | -0.02 | 0.42 | 0.00 |

Reading it:
- 1 day: RMSE skill vs climatology +5.5%, Brier skill 0.15 (beats climatology on RMSE).
- 3 day: RMSE skill vs climatology -1.6%, Brier skill 0.01 (does not beat climatology on RMSE).
- 7 day: RMSE skill vs climatology -2.1%, Brier skill -0.02 (does not beat climatology on RMSE).

Validation results point the same way as the test results at every lead. That is consistent with, though not proof of,
the model not being overfitted to one period:

| Lead (days) | Forecast | RMSE (mm) | RMSE skill vs climatology | Brier skill, rain >= 1 mm | CSI >= 2.5 mm | CSI >= 15.6 mm |
|---|---|---|---|---|---|---|
| 1 | climatology | 7.26 | 0 (reference) | - | 0.33 | 0.00 |
| 1 | persistence | 8.02 | -10.6% | - | 0.38 | 0.20 |
| 1 | xgb_hurdle | 6.72 | +7.4% | 0.22 | 0.42 | 0.04 |
| 3 | climatology | 7.26 | 0 (reference) | - | 0.33 | 0.00 |
| 3 | persistence | 9.91 | -36.6% | - | 0.25 | 0.06 |
| 3 | xgb_hurdle | 7.31 | -0.7% | 0.02 | 0.32 | 0.00 |
| 7 | climatology | 7.26 | 0 (reference) | - | 0.33 | 0.00 |
| 7 | persistence | 10.20 | -40.5% | - | 0.19 | 0.05 |
| 7 | xgb_hurdle | 7.33 | -1.0% | -0.01 | 0.32 | 0.00 |

## 3. Adding weather-model (NWP) forecasts (MOS correction)
Lead-specific archived NWP forecasts exist only from 2024, so this experiment trains on 2024 and tests on 2025
(27,375 district-days per lead). It is time-blocked, not year-blocked, and is therefore indicative, not final.

| Lead (days) | Forecast | RMSE (mm) | RMSE skill vs climatology | Correlation | Brier skill | CSI >= 2.5 mm | CSI >= 15.6 mm |
|---|---|---|---|---|---|---|---|
| 1 | climatology | 6.80 | 0 (reference) | 0.38 | - | 0.42 | 0.00 |
| 1 | persistence | 8.22 | -21.0% | 0.38 | - | 0.38 | 0.15 |
| 1 | raw_nwp | 9.27 | -36.4% | 0.49 | - | 0.45 | 0.21 |
| 1 | xgb_lag_full | 6.54 | +3.9% | 0.49 | 0.16 | 0.46 | 0.04 |
| 1 | xgb_lag_recent | 6.66 | +2.1% | 0.45 | 0.07 | 0.44 | 0.01 |
| 1 | xgb_mos | 6.15 | +9.5% | 0.59 | 0.21 | 0.49 | 0.10 |
| 3 | climatology | 6.80 | 0 (reference) | 0.38 | - | 0.42 | 0.00 |
| 3 | persistence | 9.43 | -38.7% | 0.18 | - | 0.29 | 0.07 |
| 3 | raw_nwp | 8.72 | -28.3% | 0.47 | - | 0.43 | 0.18 |
| 3 | xgb_lag_full | 6.85 | -0.7% | 0.41 | 0.02 | 0.43 | 0.00 |
| 3 | xgb_lag_recent | 7.12 | -4.7% | 0.29 | -0.11 | 0.37 | 0.00 |
| 3 | xgb_mos | 6.80 | -0.0% | 0.40 | 0.08 | 0.43 | 0.02 |
| 7 | climatology | 6.80 | 0 (reference) | 0.38 | - | 0.42 | 0.00 |
| 7 | persistence | 9.54 | -40.3% | 0.16 | - | 0.24 | 0.06 |
| 7 | raw_nwp | 7.35 | -8.1% | 0.27 | - | 0.36 | 0.03 |
| 7 | xgb_lag_full | 6.88 | -1.2% | 0.40 | -0.02 | 0.41 | 0.00 |
| 7 | xgb_lag_recent | 7.08 | -4.2% | 0.30 | -0.18 | 0.35 | 0.00 |
| 7 | xgb_mos | 7.00 | -2.9% | 0.34 | -0.09 | 0.38 | 0.00 |

Plain-language verdicts (computed, not written by hand):

- lead 1d: MOS RMSE skill vs climatology +9.5% | Brier skill +0.21 | RMSE mos 6.15, raw NWP 9.27, lag-only 6.66, clim 6.80 | CSI@15.6mm mos 0.10 vs NWP 0.21 (raw NWP is better at heavy rain) -> MOS has usable skill over climatology
- lead 3d: MOS RMSE skill vs climatology -0.0% | Brier skill +0.08 | RMSE mos 6.80, raw NWP 8.72, lag-only 7.12, clim 6.80 | CSI@15.6mm mos 0.02 vs NWP 0.18 (raw NWP is better at heavy rain) -> MOS has marginal skill over climatology
- lead 7d: MOS RMSE skill vs climatology -2.9% | Brier skill -0.09 | RMSE mos 7.00, raw NWP 7.35, lag-only 7.08, clim 6.80 | CSI@15.6mm mos 0.00 vs NWP 0.03 (raw NWP is better at heavy rain) -> MOS has no usable skill over climatology

## 4. What we can and cannot claim
- **Can claim:** for the next day, a forecast that combines past rain with weather-model output has lower error than climatology
  and than the raw weather model on district averages (2025 test). It is still worse than the raw weather model at heavy rain.
- **Cannot claim:** useful skill at 3 days from rainfall history alone, or at 7 days from any source we tested.
  Heavy rain (15.6 mm or more) is detected poorly by every smoothed forecast; the raw weather model is better at it
  than the corrected one.
- **Sanity check of the what-if baseline:** the four driest monsoons in the UP-average series are [2014, 2015, 1997, 2009]; 3 of the 3 years we expected ([2009, 2014, 2015], below-normal all-India monsoons, to be verified on IMD pages) are among them: [2009, 2014, 2015].
- **Therefore:** the dashboard labels the forecast layer as the weather model's own output and the what-if page as a
  sensitivity analysis, not a forecast.

## 5. Limits of this evidence
- The MOS experiment has one training year (2024) and one test year (2025).
- The NWP value is taken at each district's centre point while the truth is a district average, so part of the raw-NWP
  error is a mismatch of scale, not model error.
- Temperature in the what-if tool is ERA5-Land at district centres, not IMD observations.
- Heat-wave and heavy-rain thresholds follow IMD conventions as we understood them. Check them against IMD pages.
- Hyper-parameters were fixed, not tuned. We did not test whether tuning changes the picture.
