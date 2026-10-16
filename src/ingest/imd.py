"""IMD gridded rainfall loader (ticket T4).

IMD files are yearly .grd binaries (0.25 deg, -999 = missing). `scripts/download_imd.py`
caches them in data/raw/imd/. This module reads them as xarray, clipped to the UP bbox.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

UP_BBOX: dict[str, float] = {"lat_min": 23.8, "lat_max": 30.4, "lon_min": 77.1, "lon_max": 84.7}
MISSING = -900.0  # IMD uses -999; anything below this is "no data"


def clip_bbox(da: xr.DataArray, bbox: dict[str, float] = UP_BBOX, pad: float = 0.125) -> xr.DataArray:
    """Keep cells whose centre lies within the bbox (+/- pad so border cells are included)."""
    return da.sel(
        lat=slice(bbox["lat_min"] - pad, bbox["lat_max"] + pad),
        lon=slice(bbox["lon_min"] - pad, bbox["lon_max"] + pad),
    )


def clean(da: xr.DataArray) -> xr.DataArray:
    """Mask IMD missing values and impossible negatives."""
    return da.where(da > MISSING).where(lambda x: (x >= 0) | x.isnull())


def load_rain(
    start_year: int, end_year: int, root: str | Path = "data/raw/imd", bbox: dict[str, float] = UP_BBOX
) -> xr.DataArray:
    """Daily rainfall (mm/day) over the UP bbox, dims (time, lat, lon). Needs cached years."""
    import imdlib as imd  # imported lazily so unit tests do not need the package

    ds = imd.open_data("rain", start_year, end_year, "yearwise", str(root)).get_xarray()
    return clip_bbox(clean(ds["rain"]), bbox).astype("float32").rename("rain_mm")
