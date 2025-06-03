import pandas as pd
import pytest
from pathlib import Path
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
        webbrowser.open(f"file://{temp_output_file}")



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

    if open_report:
        webbrowser.open(f"file://{temp_output_file}")