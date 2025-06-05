import pytest
import pandas as pd
from pathlib import Path
from parq_tools.utils.profile_utils import ColumnarProfileReport
import types

class DummyProfile:
    def to_html(self):
        return "<html>dummy</html>"

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

def test_show_notebook(monkeypatch):
    pytest.importorskip("IPython")
    report = ColumnarProfileReport(iter([]))
    report.report = DummyProfile()
    called = {}
    def fake_display(obj):
        called['display'] = obj
    monkeypatch.setattr("IPython.display.display", fake_display)
    report.show(notebook=True)
    assert 'display' in called

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
        dummy = types.SimpleNamespace()
        dummy.to_html = lambda: "<html>dummy</html>"
        dummy.df = df
        dummy.description_set = types.SimpleNamespace(alerts=[])
        return dummy
    monkeypatch.setattr("ydata_profiling.ProfileReport", dummy_profile_report)
    report = ColumnarProfileReport(iter(cols), column_count=3, batch_size=2, show_progress=False)
    report.profile()
    assert report.report is not None
    assert hasattr(report.report, 'to_html')

