import pandas as pd
import pytest
from pathlib import Path

from parq_tools import rename_and_update_metadata
from parq_tools.parq_profile import ParquetProfileReport
import webbrowser


@pytest.fixture
def temp_parquet_file(tmp_path: Path):
    """Fixture to create a temporary Parquet file."""
    data = {
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
        "col3": [True, False, True],
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


@pytest.fixture
def temp_output_file(tmp_path):
    """Fixture to create a temporary output HTML file."""
    return tmp_path / "report.html"


@pytest.mark.parametrize("open_report", [True])
def test_native_profile_report(temp_parquet_file, temp_output_file, open_report):
    from ydata_profiling import ProfileReport
    df = pd.read_parquet(temp_parquet_file)
    report = ProfileReport(df, title="Parquet Profile Report", explorative=True, minimal=True)
    print(dir(report))
    print(report.get_description())
    report.to_file(temp_output_file)
    assert temp_output_file.exists(), "HTML report was not created."
    with open(temp_output_file, "r", encoding="utf-8") as f:
        report_content = f.read()
    assert "col1" in report_content
    assert "col2" in report_content
    assert "col3" in report_content

    if open_report:
        import webbrowser
        import unittest.mock
        with unittest.mock.patch("webbrowser.open") as mock_open:
            webbrowser.open(f"file://{temp_output_file}")
            mock_open.assert_called_once_with(f"file://{temp_output_file}")


@pytest.mark.parametrize("open_report", [True])
def test_parquet_profile_report(temp_parquet_file, temp_output_file, open_report):
    profiler = ParquetProfileReport(
        parquet_path=temp_parquet_file,
        batch_size=1,
        show_progress=False,
    )
    profiler.profile().save_html(temp_output_file)

    assert temp_output_file.exists(), "HTML report was not created."

    with open(temp_output_file, "r", encoding="utf-8") as f:
        report_content = f.read()
    assert "col1" in report_content
    assert "col2" in report_content
    assert "col3" in report_content

    # Only mock browser opening, not report generation
    if open_report:
        import webbrowser
        import unittest.mock
        with unittest.mock.patch("webbrowser.open") as mock_open:
            webbrowser.open(f"file://{temp_output_file}")
            mock_open.assert_called_once_with(f"file://{temp_output_file}")


def test_parquet_profile_report_supplied_metadata(temp_parquet_file):
    from ydata_profiling import ProfileReport
    metadata = {
        "description": "This is a test dataset",
        "version": "1.0.0",  # not a supported key in ydata_profiling
    }
    profiler = ParquetProfileReport(
        parquet_path=temp_parquet_file,
        batch_size=1,
        show_progress=False,
        dataset_metadata=metadata,
        column_descriptions={'col1': 'Column 1 description'}
    )
    profiler.profile()

    # Check if metadata is included in the report
    assert profiler.report.config.dataset.description == metadata["description"]
    assert "version" not in profiler.report.config.dataset.__dict__.keys()
    assert profiler.report.config.variables.descriptions['col1'] == "Column 1 description"


def test_parquet_profile_report_metadata_from_file(temp_parquet_file):
    from ydata_profiling import ProfileReport
    # Create a Parquet file with metadata
    rename_and_update_metadata(
        input_path=temp_parquet_file,
        output_path=temp_parquet_file,
        table_metadata={"description": "Test dataset with persisted metadata"},
        column_metadata={"col1": {"description": "Column 1 description from file"}},
        show_progress=False)
    profiler = ParquetProfileReport(
        parquet_path=temp_parquet_file,
        batch_size=1,
        show_progress=False,
    )
    profiler.profile()

    # Check if metadata is included in the report
    assert profiler.report.config.dataset.description == "Test dataset with persisted metadata"
    assert profiler.report.config.variables.descriptions['col1'] == "Column 1 description from file"


def test_parquet_profile_report_show(temp_parquet_file):
    profiler = ParquetProfileReport(
        parquet_path=temp_parquet_file,
        batch_size=1,
        show_progress=False,
    )
    profiler.profile()

    # Test notebook display path
    class DummyReport:
        def to_notebook_iframe(self):
            DummyReport.called = True

    DummyReport.called = False
    profiler.report = DummyReport()
    profiler.show(notebook=True)
    assert DummyReport.called, "Notebook display was not called."

    # Test browser open path
    import unittest.mock
    profiler.report = DummyReport()  # Dummy report with to_html
    profiler.report.to_html = lambda: "<html></html>"
    with unittest.mock.patch("webbrowser.open_new_tab") as mock_open:
        profiler.show(notebook=False)
        assert mock_open.called, "Browser open_new_tab was not called."
