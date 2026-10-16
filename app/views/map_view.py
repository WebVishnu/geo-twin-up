"""Map page: observed rainfall or 7-day forecast, one district selected at a time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, data


def _selected_from_event(event: object) -> int | None:
    try:
        pts = event.selection.points  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if not pts:
        return None
    loc = pts[0].get("location")
    return int(loc) if loc is not None else None


def _n(v: float) -> str:
    """Thousands separator for big values, one decimal for small ones (0.3 mm must not read as 0)."""
    return f"{v:,.0f}" if abs(v) >= 10 else f"{v:,.1f}"


def render(t) -> None:
    gj, names = data.load_geo()
    rain = data.load_rain()
    fc, issued = data.load_forecast()

    # Controls share one row so the map (the main thing) is visible without scrolling.
    c_layer, c_period = st.columns([1.1, 1], gap="large")
    with c_layer:
        layer = st.radio(
            t("layer"),
            ["obs", "fc"],
            format_func=lambda k: t("layer_obs") if k == "obs" else t("layer_fc"),
            horizontal=True,
        )

    if layer == "obs":
        if rain.empty:
            st.info(t("no_obs"))
            return
        last = pd.to_datetime(rain["date"]).max()
        # Default to the latest full monsoon: "the last 30 days" of a dry winter shows 0 mm everywhere.
        monsoon_year = last.year if last.month >= 9 else last.year - 1
        with c_period:
            span = st.selectbox(
                t("period"),
                ["monsoon", 30, 7, 14, 60, 90],
                format_func=lambda n: t("period_monsoon", y=monsoon_year) if n == "monsoon" else t("period_days", n=n),
            )
        dts = pd.to_datetime(rain["date"])
        if span == "monsoon":
            win = rain[(dts.dt.year == monsoon_year) & dts.dt.month.between(6, 9)]
        else:
            win = rain[dts > last - pd.Timedelta(days=int(span))]
        agg = (
            win.groupby(["district_id", "district"], as_index=False)["rain_mm"]
            .sum()
            .rename(columns={"rain_mm": "value"})
        )
        unit_title, caption = t("rain_mm"), t("data_through", d=last.date())
        daily = lambda did: (win[win.district_id == did]["date"], win[win.district_id == did]["rain_mm"])
    else:
        if fc.empty:
            st.info(t("no_forecast"))
            return
        f7 = fc[fc.lead_day.between(1, 7)]
        agg = (
            f7.groupby("district_id", as_index=False)["precip"]
            .sum()
            .rename(columns={"precip": "value"})
            .merge(names, on="district_id")
        )
        unit_title, caption = t("forecast_mm"), t("forecast_issued", d=issued)
        daily = lambda did: (f7[f7.district_id == did]["valid_date"], f7[f7.district_id == did]["precip"])

    agg["rank"] = agg["value"].rank(ascending=False, method="min").astype(int)
    sel = st.session_state.get("sel_district")
    # If every district has the same value (e.g. a dry fortnight: 0 mm everywhere) a colour scale and a
    # "rank 1 of 75" would be meaningless, so say so instead.
    flat = float(agg["value"].max()) == float(agg["value"].min())
    zlim = {"zmin": 0.0, "zmax": max(1.0, float(agg["value"].max()))} if flat else {}

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        if flat:
            st.info(t("no_rain_period"))
        fig = charts.district_map(gj, agg, "value", unit_title, selected=sel, **zlim)
        event = st.plotly_chart(
            fig,
            on_select="rerun",
            selection_mode="points",
            key="map",
            width="stretch",
            config={"displayModeBar": False},
        )
        clicked = _selected_from_event(event)
        if clicked is not None and clicked != sel:
            st.session_state["sel_district"] = clicked
            st.rerun()
        st.markdown(f"<div class='gt-muted'>{caption} · {t('hover_hint')}</div>", unsafe_allow_html=True)

    with right:
        order = agg.sort_values("district")["district_id"].tolist()
        label = dict(zip(names.district_id, names.district, strict=False))
        pick = st.selectbox(
            t("pick_district"),
            options=[None, *order],
            index=0 if sel is None else order.index(sel) + 1 if sel in order else 0,
            format_func=lambda i: "-" if i is None else label[i],
        )
        if pick != sel:
            st.session_state["sel_district"] = pick
            st.rerun()
        mean_all = float(agg["value"].mean())
        if sel is None or sel not in set(agg["district_id"]):
            st.markdown(
                f"<div class='gt-stat'><div class='v'>{_n(mean_all)}</div><div class='l'>{t('state_avg')} · {unit_title}</div></div>",
                unsafe_allow_html=True,
            )
            st.caption(t("no_district"))
            return
        row = agg[agg.district_id == sel].iloc[0]
        st.markdown(f"### {row['district']}")
        st.markdown(
            f"<div class='gt-stat'><div class='v'>{_n(float(row['value']))}</div><div class='l'>{t('total')} · {unit_title}</div></div>",
            unsafe_allow_html=True,
        )
        rank_txt = "" if flat else f"{t('rank', r=int(row['rank']), n=len(agg))} · "
        st.markdown(
            f"<div class='gt-muted'>{rank_txt}{t('state_avg')}: {_n(mean_all)}</div>",
            unsafe_allow_html=True,
        )
        d, v = daily(sel)
        st.markdown(f"**{t('daily_rain')}**")
        st.plotly_chart(charts.daily_bars(d, v, t("rain_mm")), width="stretch", config={"displayModeBar": False})
        if layer == "fc":
            r = fc[(fc.district_id == sel) & fc.lead_day.between(1, 7)]
            st.caption(t("temp_range", lo=float(r.tmin.min()), hi=float(r.tmax.max())))
