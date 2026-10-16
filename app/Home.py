"""Geo Twin AI for UP: dashboard entry point.  Run:  streamlit run app/Home.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app import style
from app.i18n import LANGS, make_t
from app.views import map_view, method_view, whatif_view

st.set_page_config(page_title="Geo Twin AI · UP", page_icon="🌧️", layout="wide")
style.inject()

with st.sidebar:
    lang = st.radio("Language / भाषा", list(LANGS), format_func=LANGS.get, horizontal=True, key="lang")
    t = make_t(lang)
    page = st.radio(
        "nav", ["map", "whatif", "method"], format_func=lambda k: t(f"nav_{k}"), label_visibility="collapsed"
    )
    st.markdown(f"<div class='gt-muted'>{t('credits')}</div>", unsafe_allow_html=True)

st.title(t("app_title"))
st.markdown(f"<p class='gt-tagline'>{t('tagline')}</p>", unsafe_allow_html=True)

{"map": map_view, "whatif": whatif_view, "method": method_view}[page].render(t)
