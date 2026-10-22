from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "Home.py")


def test_app_starts_and_shows_notice():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("not a diagnostic tool" in w.value for w in at.warning)
    assert at.dataframe  # synthetic example feature table renders
