"""Cached data access for the dashboard. All inputs are pre-aggregated files; nothing is fetched live."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parent.parent
GEO = ROOT / "data" / "external" / "up_districts.geojson"
APP_DATA = ROOT / "app_data"  # small files written by scripts/export_app_data.py; this is all the app reads


def _clockwise(geom: BaseGeometry) -> BaseGeometry:
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([orient(p, sign=-1.0) for p in geom.geoms])
    return orient(geom, sign=-1.0)


@st.cache_data(show_spinner=False)
def load_geo() -> tuple[dict[str, Any], pd.DataFrame]:
    """GeoJSON dict (for Plotly) plus a plain table of district_id/district."""
    gj = json.loads(GEO.read_text(encoding="utf-8"))
    # Plotly draws with d3-geo, which wants exterior rings clockwise (opposite of RFC 7946).
    # Wrong winding fills "the whole globe minus the polygon" instead of the district.
    for f in gj["features"]:
        f["geometry"] = mapping(_clockwise(shape(f["geometry"])))
        f["properties"]["district_id"] = int(f["properties"]["district_id"])
    names = pd.DataFrame(
        {
            "district_id": [f["properties"]["district_id"] for f in gj["features"]],
            "district": [f["properties"]["district"] for f in gj["features"]],
        }
    )
    return gj, names


@st.cache_data(show_spinner=False)
def load_rain() -> pd.DataFrame:
    """Daily district rain [district_id, district, date, rain_mm]."""
    p = APP_DATA / "rain.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["district_id", "district", "date", "rain_mm"])
    _, names = load_geo()
    df = pd.read_parquet(p).merge(names, on="district_id", how="left")
    return df.sort_values(["district_id", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_forecast() -> tuple[pd.DataFrame, str | None]:
    """Latest forecast per district: [district_id, valid_date, lead_day, precip, tmax, tmin] and its issue date."""
    p = APP_DATA / "forecast.parquet"
    if not p.exists():
        return pd.DataFrame(), None
    fc = pd.read_parquet(p)
    return fc.drop(columns=["issue_date"]), str(pd.to_datetime(fc["issue_date"].iloc[0]).date())


@st.cache_data(show_spinner=False)
def load_temp() -> pd.DataFrame:
    """Daily district tmax/tmin (ERA5-Land at district centre points): [district_id, date, tmax, tmin]."""
    p = APP_DATA / "temp.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["district_id", "date", "tmax", "tmin"])
    return pd.read_parquet(p)


def temp_series(temp: pd.DataFrame, district_id: int | None) -> pd.Series:
    """Daily tmax for one district, or the UP average when district_id is None."""
    sub = temp if district_id is None else temp[temp.district_id == district_id]
    s = sub.groupby("date")["tmax"].mean()
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def temp_ready(
    temp: pd.DataFrame,
    rain_series: pd.Series,
    district_id: int | None = None,
    min_years: int = 30,
    n_districts: int = 75,
) -> bool:
    """Temperature is usable only if it has enough baseline years for the chosen area.

    For a single district that district must have the years. For the UP average, ALL districts must be present,
    otherwise the "UP average" would silently be the average of whichever districts happen to be downloaded.
    """
    if temp.empty:
        return False
    if district_id is None:
        if temp["district_id"].nunique() < n_districts:
            return False
    else:
        temp = temp[temp["district_id"] == district_id]
        if temp.empty:
            return False
    ty = set(pd.DatetimeIndex(pd.to_datetime(temp["date"])).year)
    ry = set(pd.DatetimeIndex(rain_series.index).year)
    return len(ty & ry) >= min_years


def synthetic_tmax(years: int = 40, seed: int = 8) -> pd.Series:
    """Demo tmax with a seasonal cycle (paired with synthetic_rain; labelled as demo in the UI)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1981-01-01", periods=365 * years, freq="D")
    return pd.Series(28 + 10 * np.sin(2 * np.pi * (idx.dayofyear - 90) / 365) + rng.normal(0, 2.0, len(idx)), index=idx)


def years_available(rain: pd.DataFrame) -> int:
    return int(pd.to_datetime(rain["date"]).dt.year.nunique()) if len(rain) else 0


def state_series(rain: pd.DataFrame) -> pd.Series:
    """UP-average daily rain."""
    s = rain.groupby("date")["rain_mm"].mean()
    s.index = pd.DatetimeIndex(s.index)
    return s


def district_series(rain: pd.DataFrame, district_id: int) -> pd.Series:
    s = rain[rain.district_id == district_id].set_index("date")["rain_mm"]
    s.index = pd.DatetimeIndex(s.index)
    return s


def synthetic_rain(years: int = 40, seed: int = 7) -> pd.Series:
    """Clearly-labelled demo series with a monsoon bump. Never presented as real."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1981-01-01", periods=365 * years, freq="D")
    season = 1 + 4 * np.exp(-((idx.dayofyear - 200) ** 2) / (2 * 35**2))
    return pd.Series(rng.gamma(0.5, 7.0, len(idx)) * season * (rng.random(len(idx)) < 0.35), index=idx)
