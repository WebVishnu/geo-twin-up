"""Method and limits page. Plain statements of what the tool is and is not."""

from __future__ import annotations

import streamlit as st


def render(t) -> None:
    st.markdown(f"## {t('method_title')}")
    st.markdown(t("method_body"))
