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


def render(t) -> None:
    gj, names = data.load_geo()
    rain = data.load_rain()
    fc, issued = data.load_forecast()

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
        span = st.select_slider(
            t("period"), options=[7, 14, 30, 60, 90], value=30, format_func=lambda n: t("period_days", n=n)
        )
        win = rain[pd.to_datetime(rain["date"]) > last - pd.Timedelta(days=span)]
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

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        fig = charts.district_map(gj, agg, "value", unit_title, selected=sel)
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="map", width="stretch")
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
                f"<div class='gt-stat'><div class='v'>{mean_all:,.0f}</div><div class='l'>{t('state_avg')} · {unit_title}</div></div>",
                unsafe_allow_html=True,
            )
            st.caption(t("no_district"))
            return
        row = agg[agg.district_id == sel].iloc[0]
        st.markdown(f"### {row['district']}")
        st.markdown(
            f"<div class='gt-stat'><div class='v'>{row['value']:,.0f}</div><div class='l'>{t('total')} · {unit_title}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='gt-muted'>{t('rank', r=int(row['rank']), n=len(agg))} · {t('state_avg')}: {mean_all:,.0f}</div>",
            unsafe_allow_html=True,
        )
        d, v = daily(sel)
        st.markdown(f"**{t('daily_rain')}**")
        st.plotly_chart(charts.daily_bars(d, v, t("rain_mm")), width="stretch", config={"displayModeBar": False})
        if layer == "fc":
            r = fc[(fc.district_id == sel) & fc.lead_day.between(1, 7)]
            st.caption(t("temp_range", lo=float(r.tmin.min()), hi=float(r.tmax.max())))
