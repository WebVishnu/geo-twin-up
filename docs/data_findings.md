# Data findings (smoke test 2026-10-16)

| Check | Result | Note |
|---|---|---|
| IMD rain archive (imdlib) | FAIL | ReadTimeout: HTTPSConnectionPool(host='imdpune.gov.in', port=443): Read timed out. (read timeout=None) |
| imdlib real-time function present | PASS | get_real_data |
| Open-Meteo forecast | PASS | 8 days returned |
| Open-Meteo archive (ERA5) | PASS | 10 days |
| Open-Meteo Previous Runs @2023 | FAIL | no data in 2023 (archive starts later; probe 2024/2025) |
| Earthdata/IMERG | FAIL | set EARTHDATA_USER and EARTHDATA_PASS env vars to test (skipped) |
| ECMWF open data endpoint reachable | PASS | HTTP 200 |

## Decisions to record
- imdlib real-time lag: 
- NWP training archive length (Previous Runs): 
- MOSDAC application submitted on: 
