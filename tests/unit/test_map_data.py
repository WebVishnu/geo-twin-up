from __future__ import annotations

from shapely.geometry import shape

from app import data


def test_geojson_rings_are_clockwise_for_plotly() -> None:
    """Counter-clockwise exterior rings make Plotly paint the whole globe minus the district."""
    gj, names = data.load_geo()
    assert len(gj["features"]) == len(names) == 75
    for f in gj["features"]:
        g = shape(f["geometry"])
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        assert all(not p.exterior.is_ccw for p in polys), f["properties"]["district"]


def test_district_ids_are_unique_ints() -> None:
    gj, _ = data.load_geo()
    ids = [f["properties"]["district_id"] for f in gj["features"]]
    assert len(set(ids)) == len(ids) and all(isinstance(i, int) for i in ids)


def test_forecast_aggregation_covers_all_districts_when_logged() -> None:
    fc, issued = data.load_forecast()
    if fc.empty:
        return
    assert fc.district_id.nunique() == 75 and issued is not None
    assert fc.lead_day.min() >= 0 and fc.precip.notna().all()
