"""District aggregation of gridded fields (ticket T5).

Area-weighted zonal mean: each grid cell is a 0.25 deg box centred on its lat/lon; its weight
in a district is the intersected area (equal-area CRS). Cells with NaN are dropped and the
remaining weights renormalised, so a district half over missing cells still gets a value.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

EQUAL_AREA = "EPSG:6933"


def build_weights(districts: gpd.GeoDataFrame, lat: np.ndarray, lon: np.ndarray, step: float = 0.25) -> np.ndarray:
    """Return W with shape (n_districts, n_lat*n_lon): intersect area of each cell with each district (km^2)."""
    h = step / 2
    cells = gpd.GeoDataFrame(
        {"cell": np.arange(len(lat) * len(lon))},
        geometry=[box(x - h, y - h, x + h, y + h) for y in lat for x in lon],  # row-major: lat outer, lon inner
        crs="EPSG:4326",
    ).to_crs(EQUAL_AREA)
    d = districts.to_crs(EQUAL_AREA).reset_index(drop=True)
    W = np.zeros((len(d), len(cells)), dtype="float64")
    sidx = cells.sindex
    for i, geom in enumerate(d.geometry):
        for j in sidx.query(geom, predicate="intersects"):
            W[i, j] = geom.intersection(cells.geometry.iloc[j]).area / 1e6
    return W


def district_series(
    da: xr.DataArray, districts: gpd.GeoDataFrame, W: np.ndarray | None = None, name: str = "value"
) -> pd.DataFrame:
    """Aggregate a (time, lat, lon) DataArray to long table [district_id, district, date, <name>, coverage]."""
    da = da.transpose("time", "lat", "lon")
    if W is None:
        W = build_weights(districts, da["lat"].values, da["lon"].values)
    V = da.values.reshape(da.sizes["time"], -1).astype("float64")
    valid = ~np.isnan(V)
    num = np.where(valid, V, 0.0) @ W.T  # (T, D)
    den = valid.astype("float64") @ W.T  # (T, D) area with data
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan)
    total = W.sum(axis=1)  # (D,) district area covered by the grid box
    cov = np.where(total > 0, den / total, 0.0)  # fraction of district area with valid data
    dates = pd.to_datetime(da["time"].values)
    out = pd.DataFrame(
        {
            "district_id": np.tile(districts["district_id"].values, len(dates)),
            "district": np.tile(districts["district"].values, len(dates)),
            "date": np.repeat(dates, len(districts)),
            name: mean.reshape(-1),
            "coverage": cov.reshape(-1),
        }
    )
    return out
