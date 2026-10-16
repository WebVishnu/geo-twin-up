import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from src.geo import zonal
from src.ingest import imd


def make_districts():
    # two districts side by side covering lon 80-81 and 81-82, lat 26-27
    return gpd.GeoDataFrame(
        {"district_id": [0, 1], "district": ["A", "B"]},
        geometry=[box(80, 26, 81, 27), box(81, 26, 82, 27)],
        crs="EPSG:4326",
    )


def make_grid(value, ntime=3):
    lat = np.arange(26.125, 27.0, 0.25)   # cell centres of 0.25 deg boxes covering 26-27
    lon = np.arange(80.125, 82.0, 0.25)
    data = np.full((ntime, len(lat), len(lon)), value, dtype="float64")
    return xr.DataArray(data, coords={"time": pd.date_range("2023-01-01", periods=ntime), "lat": lat, "lon": lon}, dims=("time", "lat", "lon"))


def test_constant_field_preserved():
    out = zonal.district_series(make_grid(5.0), make_districts(), name="rain")
    assert np.allclose(out.rain, 5.0)
    assert np.allclose(out.coverage, 1.0)


def test_each_district_weight_matches_its_area():
    d = make_districts()
    da = make_grid(1.0)
    W = zonal.build_weights(d, da.lat.values, da.lon.values)
    areas = d.to_crs(zonal.EQUAL_AREA).area.values / 1e6
    assert np.allclose(W.sum(axis=1), areas, rtol=1e-3)  # cells tile the districts exactly


def test_gradient_gives_different_districts():
    da = make_grid(0.0)
    da.values[:] = (da.lon.values[None, None, :] > 81).astype(float) * 10  # east half = 10
    out = zonal.district_series(da, make_districts(), name="rain")
    a = out[out.district == "A"].rain.mean(); b = out[out.district == "B"].rain.mean()
    assert a < 1e-6 and abs(b - 10) < 1e-6


def test_nan_cells_are_dropped_not_zeroed():
    da = make_grid(4.0)
    da.values[:, :, :2] = np.nan  # west half of district A has no data
    out = zonal.district_series(da, make_districts(), name="rain")
    a = out[out.district == "A"]
    assert np.allclose(a.rain, 4.0)            # mean of valid cells, not pulled to 0
    assert np.allclose(a.coverage, 0.5, atol=0.02)


def test_imd_clean_masks_missing():
    da = xr.DataArray(np.array([-999.0, 0.0, 3.2]), dims="x")
    out = imd.clean(da)
    assert out.isnull().values.tolist() == [True, False, False]
