"""Cached data access for the dashboard. All inputs are pre-aggregated files; nothing is fetched live."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parent.parent
GEO = ROOT / "data" / "external" / "up_districts.geojson"
PROCESSED = ROOT / "data" / "processed"
FORECAST_DIR = ROOT / "data" / "raw" / "forecast_log"


def _clockwise(geom: BaseGeometry) -> BaseGeometry:
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([orient(p, sign=-1.0) for p in geom.geoms])
    return orient(geom, sign=-1.0)


@st.cache_data(show_spinner=False)
def load_geo() -> tuple[dict, pd.DataFrame]:
    """GeoJSON dict (for Plotly) plus a plain table of district_id/district."""
    gdf = gpd.read_file(GEO)
    # Plotly draws with d3-geo, which wants exterior rings clockwise (opposite of RFC 7946).
    # Wrong winding fills "the whole globe minus the polygon" instead of the district.
    gdf["geometry"] = gdf.geometry.apply(_clockwise)
    gj = json.loads(gdf.to_json())
    return gj, pd.DataFrame({"district_id": gdf["district_id"].astype(int), "district": gdf["district"]})


@st.cache_data(show_spinner=False)
def load_rain() -> pd.DataFrame:
    """Daily district rain [district_id, district, date, rain_mm]. Prefers the multi-year file."""
    files = sorted(PROCESSED.glob("district_rain_*.parquet"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        return pd.DataFrame(columns=["district_id", "district", "date", "rain_mm"])
    frames = [pd.read_parquet(f, columns=["district_id", "district", "date", "rain_mm"]) for f in files]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["district_id", "date"])
    return df.sort_values(["district_id", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_forecast() -> tuple[pd.DataFrame, str | None]:
    """Latest logged forecast aggregated to districts: [district_id, valid_date, lead_day, precip, tmax, tmin]."""
    files = sorted(FORECAST_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame(), None
    fc = pd.read_parquet(files[-1])
    gdf = gpd.read_file(GEO)[["district_id", "geometry"]]
    pts = fc[["grid_lat", "grid_lon"]].drop_duplicates().reset_index(drop=True)
    pg = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.grid_lon, pts.grid_lat), crs="EPSG:4326")
    inside = gpd.sjoin(pg, gdf, predicate="within")[["grid_lat", "grid_lon", "district_id"]]
    # Small districts may contain no grid point: give them the nearest one.
    missing = gdf[~gdf.district_id.isin(inside.district_id)].copy()
    if len(missing):
        missing["geometry"] = missing.geometry.representative_point()
        near = gpd.sjoin_nearest(missing, pg, how="left")[["district_id", "grid_lat", "grid_lon"]]
        inside = pd.concat([inside, near], ignore_index=True)
    m = fc.merge(inside, on=["grid_lat", "grid_lon"])
    agg = (
        m.groupby(["district_id", "valid_date", "lead_day"], as_index=False)
        .agg(precip=("precip", "mean"), tmax=("tmax", "mean"), tmin=("tmin", "mean"))
        .sort_values(["district_id", "valid_date"])
    )
    return agg, str(pd.to_datetime(fc["issue_date"].iloc[0]).date())


@st.cache_data(show_spinner=False)
def load_temp() -> pd.DataFrame:
    """Daily district tmax/tmin (ERA5-Land at district centre points): [district_id, date, tmax, tmin]."""
    p = PROCESSED / "district_temp_all.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["district_id", "date", "tmax", "tmin"])
    return pd.read_parquet(p, columns=["district_id", "date", "tmax", "tmin"])


def temp_series(temp: pd.DataFrame, district_id: int | None) -> pd.Series:
    """Daily tmax for one district, or the UP average when district_id is None."""
    sub = temp if district_id is None else temp[temp.district_id == district_id]
    s = sub.groupby("date")["tmax"].mean()
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def temp_ready(temp: pd.DataFrame, rain_series: pd.Series, min_years: int = 30) -> bool:
    """Temperature is usable only if it spans (almost) the same years as rainfall and has enough baseline."""
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
