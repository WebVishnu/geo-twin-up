# Deploying the dashboard (Streamlit Community Cloud, free)

The deployed app reads only small files in `app_data/` and `data/external/up_districts.geojson`, and installs only
`app/requirements.txt` (no geopandas, xgboost or imdlib). It was checked in a clean virtual environment built from
that file alone.

## One-time steps (you, in the browser)
1. Open https://share.streamlit.io and sign in with GitHub. Allow access to the `geo-twin-up` repository.
2. Click **Create app** > **Deploy a public app from GitHub**.
3. Repository: `WebVishnu/geo-twin-up`. Branch: `main`. Main file path: `app/Home.py`.
4. Open **Advanced settings** and choose Python **3.11** or **3.12**.
5. Click **Deploy**. The first build takes a few minutes. Copy the public URL.
6. Optional: pick a custom subdomain in the app settings, for example `geo-twin-up`.

## Keeping it fresh
- The GitHub Action `daily-forecast-log` runs at 07:00 IST, logs the forecast, exports `app_data/forecast.parquet`
  and pushes. A push triggers a redeploy, so the forecast layer updates daily.
- Rain and temperature history change rarely. When new IMD years arrive:
  ```powershell
  python scripts/download_imd.py 1991 2025
  python scripts/build_district_data.py
  python scripts/download_temperature.py 1991 2025     # resumable; re-run after API limits reset
  python scripts/export_app_data.py
  git add app_data; git commit -m "data: refresh history"; git push
  ```

## Limits to know
- Free apps sleep after a period of no visits and wake in about a minute.
- Open-Meteo's free tier is non-commercial. Keep the app as an academic prototype.
- Until 30+ years of IMD rain are exported, the what-if page shows a labelled synthetic demo only.
