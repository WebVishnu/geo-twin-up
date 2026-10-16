from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.i18n import LANGS, STR, make_t

APP = str(Path(__file__).resolve().parents[2] / "app" / "Home.py")


def run(lang: str, page_index: int) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(lang).run()
    at.sidebar.radio[1].set_value(["map", "whatif", "method"][page_index]).run()
    return at


def test_language_keys_match() -> None:
    assert set(STR["en"]) == set(STR["hi"])
    assert set(LANGS) == set(STR)


def test_no_untranslated_hindi_strings() -> None:
    same = [k for k in STR["en"] if STR["en"][k] == STR["hi"][k] and k not in {"app_title"}]
    assert same == []


def test_format_placeholders_match() -> None:
    import re

    for k in STR["en"]:
        assert sorted(re.findall(r"{(\w+)", STR["en"][k])) == sorted(re.findall(r"{(\w+)", STR["hi"][k])), k


def test_translator_raises_on_unknown_key() -> None:
    with pytest.raises(KeyError):
        make_t("en")("nope")


@pytest.mark.parametrize("lang", ["en", "hi"])
@pytest.mark.parametrize("page", [0, 1, 2])
def test_every_page_renders_in_both_languages(lang: str, page: int) -> None:
    at = run(lang, page)
    assert not at.exception
    assert at.title[0].value == STR[lang]["app_title"]


def test_whatif_demo_mode_is_labelled_and_works(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.data as d

    monkeypatch.setattr(d, "years_available", lambda _r: 3)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[1].set_value("whatif").run()
    assert not at.exception
    assert any("30" in i.value for i in at.info)  # explains the 30-year requirement
    at.button[0].click().run()  # opt in to demo data
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "gt-demo" in html and "Sensitivity analysis" in html
