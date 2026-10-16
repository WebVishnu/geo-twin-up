"""What-if page: perturb observed history and re-score the monsoon. Never uses the forecast models."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, data
from src.indices import spi
from src.whatif.engine import Scenario, WhatIfEngine

MIN_YEARS = spi.MIN_YEARS
INDICATORS = ["jjas_ratio_to_lpa", "spi3_monsoon_mean", "rx1day_mm", "rx5day_mm", "heavy_rain_days", "heatwave_days"]


@st.cache_resource(show_spinner="...")
def _engine(key: str, series: pd.Series) -> WhatIfEngine:
    yrs = sorted(set(series.index.year))
    return WhatIfEngine(series, None, baseline_years=(yrs[0], yrs[-1]), n_boot=150)


def _fmt(name: str, v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:.0%}" if name == "jjas_ratio_to_lpa" else f"{v:.1f}" if abs(v) < 100 else f"{v:,.0f}"


def render(t) -> None:
    st.markdown(f"## {t('whatif_title')}")
    st.caption(t("whatif_intro"))

    rain = data.load_rain()
    have = data.years_available(rain)
    demo = st.session_state.get("demo", False)
    if have < MIN_YEARS and not demo:
        st.info(t("need_years", n=MIN_YEARS, have=have))
        if st.button(t("demo_btn")):
            st.session_state["demo"] = True
            st.rerun()
        return

    st.markdown(f"<div class='gt-banner'>{t('banner')}</div>", unsafe_allow_html=True)
    if demo and have < MIN_YEARS:
        st.markdown(f"<div class='gt-banner gt-demo'>{t('demo_banner')}</div>", unsafe_allow_html=True)

    _, names = data.load_geo()
    label = dict(zip(names.district_id, names.district, strict=False))
    area = st.selectbox(
        t("area"), [0, *sorted(label, key=label.get)], format_func=lambda i: t("all_up") if i == 0 else label[i]
    )
    if demo and have < MIN_YEARS:
        series, key = data.synthetic_rain(), "demo"
    elif area == 0:
        series, key = data.state_series(rain), "all"
    else:
        series, key = data.district_series(rain, area), f"d{area}"

    st.session_state.setdefault("dp", 0)
    st.session_state.setdefault("ex", 0)
    st.session_state.setdefault("tp", 0.0)
    p1, p2, p3, _ = st.columns([1, 1, 1, 3])
    if p1.button(t("preset_dry20")):
        st.session_state["dp"] = -20
    if p2.button(t("preset_dry30")):
        st.session_state["dp"] = -30
    if p3.button(t("preset_reset")):
        st.session_state.update(dp=0, ex=0, tp=0.0)

    c1, c2, c3 = st.columns(3)
    dp = c1.slider(t("rain_change"), -50, 50, key="dp", step=5)
    ex = c2.slider(t("heavy_change"), -30, 50, key="ex", step=5)
    tp = c3.slider(t("temp_change"), -5.0, 5.0, key="tp", step=0.5, disabled=True)
    st.caption(t("temp_missing"))
    del tp

    eng = _engine(key, series)
    res = eng.run(Scenario(rain_factor=dp / 100, extreme_factor=ex / 100, temp_delta=0.0))

    years = [int(y) for y in res.baseline.index]
    fig = charts.ratio_chart(
        years,
        (res.baseline["jjas_ratio_to_lpa"] * 100).round(1).tolist(),
        (res.perturbed["jjas_ratio_to_lpa"] * 100).round(1).tolist(),
        t("line75"),
        t("before"),
        t("after"),
    )
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown(f"**{t('chart_ratio')}**")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        b, a = res.drought_share_baseline, res.drought_share_scenario
        st.markdown(
            f"<div class='gt-stat'><div class='v'>{a:.0%}</div><div class='l'>{t('drought_share')} · {t('before')} {b:.0%}</div></div>",
            unsafe_allow_html=True,
        )
        rows = [
            "<div class='gt-row gt-head'><span class='n'></span><span class='b'>"
            + t("before")
            + "</span><span class='a'>"
            + t("after")
            + "</span><span class='d'>"
            + t("change")
            + "</span></div>"
        ]
        for name in INDICATORS:
            if name not in res.summary.index:
                continue
            r = res.summary.loc[name]
            rows.append(
                f"<div class='gt-row'><span class='n'>{t('ind_' + name)}</span><span class='b'>{_fmt(name, r['baseline'])}</span>"
                f"<span class='a'>{_fmt(name, r['scenario'])}</span><span class='d'>{_fmt(name, r['delta'])}</span></div>"
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption(t("ci_note"))
