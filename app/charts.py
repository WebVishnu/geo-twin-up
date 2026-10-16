"""Plotly figures: the district map and small supporting charts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from app import style


def district_map(
    gj: dict[str, Any],
    df: pd.DataFrame,
    value_col: str,
    colorbar_title: str,
    selected: int | None = None,
    scale: list[str] | None = None,
    zmid: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Borderless map of the 75 districts. No basemap: the district shapes are the map.

    Pass zmin/zmax to pin the colour range, e.g. 0..1 when every district has the same value.
    """
    fig = go.Figure(
        go.Choropleth(
            geojson=gj,
            locations=df["district_id"],
            featureidkey="properties.district_id",
            z=df[value_col],
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            colorscale=scale or style.RAIN_SCALE,
            marker_line_color="white",
            marker_line_width=0.6,
            customdata=df[["district"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>%{z:,.0f}<extra></extra>",
            colorbar={
                "orientation": "h",
                "thickness": 10,
                "len": 0.36,
                "x": 0.99,
                "xanchor": "right",
                "y": 0.99,
                "yanchor": "top",
                "title": {"text": colorbar_title, "side": "top", "font": {"size": 13, "color": style.MUTED}},
                "tickfont": {"size": 12, "color": style.MUTED},
                "outlinewidth": 0,
            },
        )
    )
    if selected is not None and selected in set(df["district_id"]):
        fig.add_trace(
            go.Choropleth(
                geojson=gj,
                locations=[selected],
                featureidkey="properties.district_id",
                z=[0],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_color=style.INK,
                marker_line_width=2.6,
                showscale=False,
                hoverinfo="skip",
            )
        )
    # Fixed frame on UP (fitbounds leaves a lot of empty margin); Mercator keeps shapes familiar at this latitude.
    fig.update_geos(
        visible=False,
        projection_type="mercator",
        lonaxis_range=[76.9, 84.9],
        lataxis_range=[23.6, 30.55],
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Mukta, sans-serif", "color": style.INK},
        dragmode=False,
        clickmode="event+select",
    )
    return fig


def daily_bars(
    dates: pd.Series, values: pd.Series, name: str, color: str = style.RIVER, height: int = 190
) -> go.Figure:
    fig = go.Figure(
        go.Bar(x=dates, y=values, marker_color=color, name=name, hovertemplate="%{x|%d %b}: %{y:.1f} mm<extra></extra>")
    )
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 6, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Mukta, sans-serif", "color": style.INK},
        xaxis={"showgrid": False, "tickformat": "%d %b"},
        yaxis={"gridcolor": style.RULE, "zeroline": False, "title": None, "rangemode": "tozero"},
        showlegend=False,
        bargap=0.15,
    )
    return fig


def ratio_chart(
    years: list[int], base: list[float], scen: list[float], line_label: str, before: str, after: str, height: int = 330
) -> go.Figure:
    """Monsoon rain as % of normal, per year: before vs after, with the 75% drought line."""
    fig = go.Figure()
    fig.add_bar(x=years, y=base, name=before, marker_color="#BFDCDB")
    fig.add_bar(x=years, y=scen, name=after, marker_color=style.RIVER)
    # The drought line is a legend entry, not a text label: a label on the plot collides with the bars.
    fig.add_scatter(
        x=[years[0] - 0.5, years[-1] + 0.5],
        y=[75, 75],
        mode="lines",
        name=line_label,
        line={"color": style.OCHRE, "width": 2.5},
        hoverinfo="skip",
    )
    fig.update_layout(
        barmode="group",
        height=height,
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Mukta, sans-serif", "color": style.INK},
        xaxis={"showgrid": False, "dtick": 5},
        yaxis={"gridcolor": style.RULE, "zeroline": False, "ticksuffix": "%"},
        legend={"orientation": "h", "y": -0.18},
        bargap=0.2,
    )
    return fig
