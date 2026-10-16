# Claims audit: synopsis vs what we can defend

Purpose: list every claim in the synopsis (`GEO_TW_1.DOC`) that is wrong, unverified or too strong, with the wording
we recommend instead. Status as of October 2026. "Verify" means we could not confirm it from a primary source.

| # | Synopsis says | Status | Say this instead |
|---|---|---|---|
| 1 | No system, Indian or international, at any scale, combines these data into a digital twin with what-if | **Not defensible.** IMD launched an AI-based 1 km, 10-day rainfall forecast pilot for Uttar Pradesh in May 2026 (news summaries; primary PIB page not found, verify). Destination Earth already offers what-if runs. | "To our knowledge, based on public sources up to October 2026, we found no openly accessible, UP-specific, district-level tool that combines open forecast data, hazard indices and interactive what-if sensitivity analysis in one application." Avoid "first", "only", "no system". |
| 2 | IMD real-time API has new registrations on hold | **Unverified.** No source found. What is documented: the public AWS/ARG portal was locked around May 2025 and users were pointed to a paid data-supply portal. | State only what you can screenshot from the IMD portal on a dated visit. |
| 3 | MOSDAC general users get Level-1 data with about 3 days latency | **Unverified.** | Quote the exact sentence from the MOSDAC manual or drop the number. |
| 4 | ConvLSTM-XGBoost study [9] supports the ensemble | **Misleading.** The study is about Yunnan, China, monthly data; its abstract has an unfilled template placeholder and a CMIP6 claim with no results in the body. | "A study in China showing a hybrid ConvLSTM-XGBoost idea." Do not use it as India evidence. |
| 5 | South Asia benchmark [10] motivates the XGBoost + LSTM ensemble | **Overstated.** Monthly values, 1901-2023. It does not test daily 1-7 day forecasts. | "A monthly-scale comparison." |
| 6 | Springer 2024 XGBoost study [8] | **Not read in full.** Seasonal all-India rainfall. | Cite as indicative; quote no metrics we have not read. |
| 7 | IMD temperature archive 1901-present | **Wrong in detail.** Daily Tmax/Tmin grids start in 1951 and are 1.0 degree (about 100 km). | State 1951 and 1.0 degree. We therefore use ERA5-Land at district centres for temperature and say so. |
| 8 | Bharat Forecast System cited to Wikipedia [2] | **Weak source.** | Cite PIB or IITM. Reported: 6 km, Arka supercomputer 11.77 petaflops, adopted by IMD 26 May 2025. |
| 9 | GAEA "still has not delivered what-if" [7] | **Too strong.** The full text lists interactive what-if as a user request and planned extension, and mentions IPCC-scenario risk views. | "GAEA identifies interactive what-if simulation as an open challenge." Check the author list in ref [7] against arXiv 2511.13807. |
| 10 | Pipeline loops back to re-tune features if validation is "not acceptable" | **Leakage risk.** Tuning against test data inflates results. | Tune on a validation period only. Score the test period once. Our code does this. |
| 11 | Fusing IMD and MOSDAC is the main contribution | **High risk.** MOSDAC needs approval and large downloads for little gain in a first prototype. | MOSDAC is a stretch goal, not a dependency. |
| 12 | XGBoost + LSTM gives useful short-term forecasts | **Our own result says: only for about 1-3 days, and only against climatology.** On the provisional test split the seasonal-reference Brier skill was 0.23 at 1 day, 0.05 at 3 days and 0.01 at 7 days; heavy-rain detection was near zero. | Report this table. Beyond about 3 days the forecast layer is the weather model's own output. |
| 13 | "Continuously updated" platform | **Not true.** It updates once a day in a batch. | "Daily batch update." Never say "real-time". |
| 14 | Digital twin | **Strictly, no.** There is no automatic two-way flow with the physical system. | "Digital-twin-style decision-support prototype." |

## What we did about each
- Items 1, 9, 13, 14: wording is fixed in the app's Method page (English and Hindi).
- Items 2, 3: not used anywhere in the app.
- Item 10: splits are year-blocked from `configs/base.yaml`; the pipeline prints PROVISIONAL when configured years are missing.
- Item 12: `reports/metrics.csv` and `reports/mos_metrics.csv` are the evidence.

## Still to verify by hand (do before submission)
- The IMD UP AI pilot on a PIB or IMD page.
- IMD heat-wave criteria and the 64.5 mm heavy-rain class on IMD pages.
- Full text of the Springer 2024, Scientific Reports 2025 and GAEA papers before quoting numbers.
- Licence terms of IMD gridded data.
