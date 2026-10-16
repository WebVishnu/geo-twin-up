"""Aggregate a logged grid forecast to districts (used by the exporter, not by the deployed app)."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


def aggregate_forecast(fc: pd.DataFrame, districts: gpd.GeoDataFrame) -> pd.DataFrame:
    """Mean forecast per district over the grid points inside it.

    Args:
        fc: Forecast log with columns grid_lat, grid_lon, valid_date, lead_day, precip, tmax, tmin.
        districts: GeoDataFrame with district_id and geometry (EPSG:4326).

    Returns:
        Table [district_id, valid_date, lead_day, precip, tmax, tmin]. A district that contains no grid
        point (small ones) takes the nearest point, so all districts are always present.
    """
    gdf = districts[["district_id", "geometry"]]
    pts = fc[["grid_lat", "grid_lon"]].drop_duplicates().reset_index(drop=True)
    pg = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.grid_lon, pts.grid_lat), crs="EPSG:4326")
    inside = gpd.sjoin(pg, gdf, predicate="within")[["grid_lat", "grid_lon", "district_id"]]
    missing = gdf[~gdf.district_id.isin(inside.district_id)].copy()
    if len(missing):
        missing["geometry"] = missing.geometry.representative_point()
        near = gpd.sjoin_nearest(missing, pg, how="left")[["district_id", "grid_lat", "grid_lon"]]
        inside = pd.concat([inside, near], ignore_index=True)
    merged = fc.merge(inside, on=["grid_lat", "grid_lon"])
    return (
        merged.groupby(["district_id", "valid_date", "lead_day"], as_index=False)
        .agg(precip=("precip", "mean"), tmax=("tmax", "mean"), tmin=("tmin", "mean"))
        .sort_values(["district_id", "valid_date"])
        .reset_index(drop=True)
    )
