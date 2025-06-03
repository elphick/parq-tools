from pathlib import Path
from typing import Iterator, Optional, List, Union
import pandas as pd
import pyarrow.parquet as pq

from ydata_profiling import ProfileReport

from parq_tools.utils.profile_utils import ColumnarProfileReport


def parquet_column_generator(
        parquet_path: Union[str, Path],
        columns: Optional[List[str]] = None
) -> Iterator[pd.Series]:
    """
    Yields columns from a Parquet file as pandas Series.

    Args:
        parquet_path (str or Path): Path to the Parquet file.
        columns (List[str], optional): List of column names to yield. If None, yields all columns.

    Yields:
        pd.Series: Each column as a pandas Series.
    """
    pq_file = pq.ParquetFile(str(parquet_path))
    all_columns = columns or pq_file.schema.names
    for col in all_columns:
        series = pq_file.read(columns=[col]).to_pandas()[col]
        yield series


class ParquetProfileReport:
    """
    High-level profiler for Parquet files using ColumnarProfileReport.
    """

    def __init__(
            self,
            parquet_path: Union[str, Path],
            columns: Optional[List[str]] = None,
            batch_size: Optional[int] = 1,  # Number of columns to process in each batch
            show_progress: bool = True,
    ):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.report: Optional[ProfileReport] = None

        if columns is None:
            pq_file = pq.ParquetFile(str(self.parquet_path))
            self.columns = pq_file.schema.names
        else:
            self.columns = columns

    def profile(self) -> 'ParquetProfileReport':
        """
        Profiles the Parquet file

        """
        if self.batch_size is None:
            # Native ydata profiling (no chunking)
            df = pd.read_parquet(self.parquet_path, columns=self.columns)
            self.report = ProfileReport(df, minimal=True, explorative=False, progress_bar=False)
        else:
            # Columnar profiling
            gen = parquet_column_generator(self.parquet_path, columns=self.columns)
            report = ColumnarProfileReport(
                column_generator=gen,
                column_count=len(self.columns),
                batch_size=self.batch_size,
                show_progress=self.show_progress,
            )
            report.profile()
            self.report = report.report
        return self

    def to_html(self) -> str:
        if self.report is None:
            raise RuntimeError("No report generated. Call profile() first.")
        return self.report.to_html()

    def save_html(self, output_html: Path) -> None:
        output_html.write_text(self.to_html(), encoding="utf-8")

    def show(self, notebook: bool = False):
        """
        Display the profile report in a notebook or open in a browser.

        Args:
            notebook (bool): If True, display in Jupyter notebook. If False, open in browser.
        """
        if notebook:
            self.report.to_notebook_iframe()
        else:
            import tempfile, webbrowser
            tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
            tmp.write(self.to_html().encode("utf-8"))
            tmp.close()
            webbrowser.open_new_tab(f"file://{tmp.name}")
