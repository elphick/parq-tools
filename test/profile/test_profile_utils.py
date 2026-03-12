import pytest
import pandas as pd
from pathlib import Path
from parq_tools.utils.profile_utils import ColumnarProfileReport
import types

class DummyProfile:
    def to_html(self):
        return "<html>dummy</html>"

    def to_notebook_iframe(self):
        # Minimal stub for notebook display; real implementation would display in Jupyter.
        return "<iframe>dummy</iframe>"

def test_to_html_raises_if_no_report():
    report = ColumnarProfileReport(iter([]))
    with pytest.raises(RuntimeError, match="No report generated"):
        report.to_html()

def test_save_html_raises_if_to_html_fails(tmp_path):
    class FailingReport(ColumnarProfileReport):
        def to_html(self):
            raise RuntimeError("fail")
    failing = FailingReport(iter([]))
    with pytest.raises(RuntimeError, match="fail"):
        failing.save_html(tmp_path / "out.html")

def test_show_notebook():
    # Ensure that calling show(notebook=True) delegates to the profile's
    # to_notebook_iframe method without raising.
    report = ColumnarProfileReport(iter([]))
    dummy = DummyProfile()
    called = {"notebook": False}

    def fake_iframe():
        called["notebook"] = True
        return "<iframe>dummy</iframe>"

    dummy.to_notebook_iframe = fake_iframe
    report.report = dummy
    report.show(notebook=True)
    assert called["notebook"] is True

def test_show_browser(monkeypatch, tmp_path):
    report = ColumnarProfileReport(iter([]))
    report.report = DummyProfile()
    called_urls = []
    monkeypatch.setattr("webbrowser.open_new_tab", lambda *args, **kwargs: called_urls.append(args[0]))
    # Patch save_html to just create a file
    def fake_save_html(path):
        Path(path).write_text("<html>dummy</html>")
    report.save_html = fake_save_html
    report.show(notebook=False)
    assert called_urls

def test_profile_batches(monkeypatch):
    # Test that profile() works with a simple generator
    cols = [pd.Series([1,2,3], name=f"col{i}") for i in range(3)]
    def dummy_profile_report(df, *a, **k):
        # Minimal object that behaves like a ProfileReport for the purposes of
        # ColumnarProfileReport.profile. It must provide attributes accessed in
        # BatchDescription via report.config, report.summarizer and report.typeset,
        # as well as to_html, df and description_set.
        dummy = types.SimpleNamespace()
        dummy.to_html = lambda: "<html>dummy</html>"
        dummy.df = df
        dummy.description_set = types.SimpleNamespace(alerts=[])
        # Provide minimal config/summarizer/typeset so that BatchDescription
        # can be constructed without relying on the real ydata_profiling
        # implementation. These are not used by the dummy, but are required
        # attributes for the code path.
        dummy.config = types.SimpleNamespace()
        dummy.summarizer = types.SimpleNamespace()
        dummy.typeset = types.SimpleNamespace()
        return dummy

    # Patch the optional import helper so that profile() uses our dummy
    monkeypatch.setattr(
        "parq_tools.utils.optional_imports.get_ydata_profile_report",
        lambda feature="profiling": dummy_profile_report,
    )
    report = ColumnarProfileReport(iter(cols), column_count=3, batch_size=2, show_progress=False)
    report.profile()
    assert report.report is not None
    assert hasattr(report.report, 'to_html')

