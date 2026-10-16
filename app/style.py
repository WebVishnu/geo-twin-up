"""Visual tokens and CSS. One family (Mukta) covers Latin and Devanagari, so switching language keeps the voice."""

from __future__ import annotations

import streamlit as st

PAPER = "#F5F8F7"
INK = "#12313F"
MUTED = "#5B7380"
RULE = "#D5E0E3"
RIVER = "#2F7FA6"
OCHRE = "#B7791F"
HEAT = "#B83A2E"

# Rain: pale water to deep monsoon blue. Anomaly: dry ochre to wet blue through a near-white midpoint.
RAIN_SCALE = ["#EAF2F0", "#BFDCDB", "#8CC1C5", "#4F9DBC", "#2F7FA6", "#1F5F8B", "#10345C"]
ANOMALY_SCALE = ["#B7791F", "#E3B968", "#F3EBD9", "#BFDCDB", "#4F9DBC", "#10345C"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select {{
  font-family: 'Mukta', system-ui, sans-serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ max-width: 1280px; padding-top: 1.6rem; padding-bottom: 3rem; }}
h1 {{ font-weight: 800; letter-spacing: -0.01em; font-size: 2.1rem; line-height: 1.15; margin-bottom: .2rem; }}
h2, h3 {{ font-weight: 600; color: {INK}; }}
.gt-tagline {{ color: {MUTED}; max-width: 62ch; font-size: 1.05rem; line-height: 1.5; margin: 0 0 1.2rem 0; }}
.gt-banner {{ border-left: 4px solid {OCHRE}; background: #FBF4E4; padding: .7rem 1rem; margin: .4rem 0 1.2rem 0;
  color: {INK}; font-weight: 600; max-width: 78ch; }}
.gt-demo {{ border-left-color: {HEAT}; background: #FBE9E6; }}
.gt-stat {{ border-top: 2px solid {INK}; padding-top: .5rem; margin-bottom: 1rem; }}
.gt-stat .v {{ font-size: 2.3rem; font-weight: 800; line-height: 1; }}
.gt-stat .l {{ color: {MUTED}; font-size: .95rem; margin-top: .25rem; }}
.gt-muted {{ color: {MUTED}; font-size: .92rem; }}
.gt-row {{ display: flex; justify-content: space-between; gap: 1rem; border-bottom: 1px solid {RULE}; padding: .45rem 0; }}
.gt-row .n {{ flex: 1; }} .gt-row .b {{ color: {MUTED}; min-width: 5.5rem; text-align: right; }}
.gt-row .a {{ font-weight: 600; min-width: 5.5rem; text-align: right; }}
.gt-row .d {{ min-width: 5.5rem; text-align: right; font-weight: 600; }}
.gt-head {{ color: {MUTED}; font-size: .88rem; border-bottom: 2px solid {INK}; }}
section[data-testid="stSidebar"] {{ background: #EAF1F1; border-right: 1px solid {RULE}; }}
:focus-visible {{ outline: 3px solid {RIVER} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
