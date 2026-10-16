# geo-twin-up

Geo Twin AI for UP — a digital-twin-style climate decision-support prototype for Uttar Pradesh
(forecast + hazard indices + what-if *sensitivity analysis*). Daily batch updates, not real-time.

## Setup (Windows)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/unit -q
```

## First steps (week 1)
1. `python scripts/smoke_data.py` → writes `docs/data_findings.md` (fill in the decisions at the bottom).
2. `python -m src.ingest.forecast_logger` → logs today's Open-Meteo forecasts to `data/raw/forecast_log/`.
   The GitHub Action `daily-forecast-log` does this daily and commits the file.
3. Never commit credentials. Earthdata: set `EARTHDATA_USER` / `EARTHDATA_PASS` locally or as GitHub Secrets.

Docs: planning files are in the parent project's `reports/` folder.

## Dashboard
```powershell
python scripts/build_district_data.py   # after scripts/download_imd.py has cached some years
streamlit run app/Home.py
```
Language switch (English / हिन्दी) is in the sidebar. The what-if page needs >= 30 years of IMD data; until then it offers an explicitly labelled synthetic demo.
