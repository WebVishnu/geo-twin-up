import datetime as dt

import pandas as pd

from src.ingest import forecast_logger as fl

TODAY = dt.date(2026, 10, 17)


def fake_fetch(points):
    days = [(TODAY + dt.timedelta(days=i)).isoformat() for i in range(fl.FORECAST_DAYS)]
    return [
        {
            "latitude": lat,
            "longitude": lon,
            "daily": {
                "time": days,
                "precipitation_sum": [1.0] * len(days),
                "temperature_2m_max": [35.0] * len(days),
                "temperature_2m_min": [22.0] * len(days),
            },
        }
        for lat, lon in points
    ]


def test_grid_inside_bbox():
    g = fl.make_grid(1.0)
    assert g and all(23.8 <= a <= 30.4 and 77.1 <= o <= 84.7 for a, o in g)


def test_schema_and_leads(tmp_path):
    p = fl.run(tmp_path, step=2.0, fetch=fake_fetch, today=TODAY, sleep_s=0)
    df = pd.read_parquet(p)
    assert {"issue_date", "lead_day", "valid_date", "lat", "lon", "precip", "tmax", "tmin"} <= set(df)
    assert df.grid_lat.nunique() * df.grid_lon.nunique() >= df.groupby(["grid_lat", "grid_lon"]).ngroups
    assert len(fl.make_grid(2.0)) == df.groupby(["grid_lat", "grid_lon"]).ngroups
    assert sorted(df.lead_day.unique()) == list(range(fl.FORECAST_DAYS))
    # no leakage: valid date is never before the issue date
    assert (pd.to_datetime(df.valid_date) >= pd.Timestamp(TODAY)).all()


def test_idempotent(tmp_path):
    first = fl.run(tmp_path, step=2.0, fetch=fake_fetch, today=TODAY, sleep_s=0)
    again = fl.run(tmp_path, step=2.0, fetch=fake_fetch, today=TODAY, sleep_s=0)
    assert first is not None and again is None
    assert len(list(tmp_path.glob("*.parquet"))) == 1
