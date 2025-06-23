"""
parq_profile.py

Utilities for profiling Parquet files and generating HTML reports using ydata-profiling, with support for notebook and browser display.

Main API:

- ParquetProfileReport: Class for generating, saving, and displaying profile reports for Parquet files.
"""
import json
from pathlib import Path
from typing import Iterator, Optional, List, Union
import pandas as pd
import pyarrow.parquet as pq

from ydata_profiling import ProfileReport

from parq_tools.utils import atomic_output_file
from parq_tools.utils.metadata_utils import get_table_metadata, get_column_metadata, get_pandas_metadata
from parq_tools.utils.profile_utils import ColumnarProfileReport, ProfileMetadata


def parquet_column_generator(parquet_path: Union[str, Path],
                             columns: Optional[List[str]] = None) -> Iterator[pd.Series]:
    """
    Yields columns from a Parquet file as pandas Series.

    Args:
        parquet_path (str or Path): Path to the Parquet file.
        columns (List[str], optional): List of column names to yield. If None, yields all columns.

    Yields:
        pd.Series: Each column as a pandas Series.
    """
    pq_file = pq.ParquetFile(str(parquet_path))
    pandas_metadata = get_pandas_metadata(pq_file)
    if pandas_metadata:
        index_columns = pandas_metadata.get('index_columns', [])
    else:
        index_columns = []

    all_columns = columns or pq_file.schema.names
    for col in all_columns:
        if col not in pq_file.schema.names:
            raise ValueError(f"Column '{col}' not found in Parquet file.")
        if col in index_columns:
            series = pq_file.read(columns=[col]).to_pandas().reset_index()[col]
        else:
            series = pq_file.read(columns=[col]).to_pandas()[col]
        yield series


class ParquetProfileReport:
    """For ydata-profiler reports on large parquet files.

    Useful for profiling large Parquet files without loading them entirely into memory.
    This class supports both native profiling (without chunking) and columnar profiling (with chunking).
    """

    def __init__(self,
                 parquet_path: Union[str, Path],
                 columns: Optional[List[str]] = None,
                 batch_size: Optional[int] = 1,  # Number of columns to process in each batch
                 show_progress: bool = True,
                 title: str = "Parquet Profile Report",
                 dataset_metadata: Optional[Union[dict, ProfileMetadata]] = None,
                 column_descriptions: Optional[dict[str, str]] = None) -> None:
        """
        Initialize the ParquetProfileReport.

        Args:
            parquet_path: Path to the Parquet file to profile.
            columns: List of column names to include in the profile. If None, all columns are used.
            batch_size: Optional[int]: Number of columns to process in each batch. If None,
             processes all columns at once.
            show_progress: bool: If True, displays a progress bar during profiling.
            title: Title of the report.
            dataset_metadata: Optional[Union[dict, ProfileMetadata]]: Metadata for the dataset.  Will over-ride any
                metadata in the Parquet file.
            column_descriptions: Optional[dict[str, str]]: Column descriptions for the dataset.  Will over-ride any
                descriptions in the Parquet file.
        """
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.title = title
        self.report: Optional[ProfileReport] = None

        metadata = dataset_metadata if isinstance(dataset_metadata, ProfileMetadata) else ProfileMetadata.from_dict(
            dataset_metadata) if dataset_metadata else None
        self.dataset_metadata = metadata
        self.column_descriptions = column_descriptions

        pq_file = pq.ParquetFile(str(self.parquet_path))
        self.columns = pq_file.schema.names if columns is None else columns
        if not self.dataset_metadata:
            # If no metadata is provided, use the Parquet file metadata
            table_meta: dict = get_table_metadata(pq_file)
            self.dataset_metadata = ProfileMetadata.from_dict(table_meta) if pq_file.metadata else None
        if self.column_descriptions is None:
            # If no column descriptions are provided, use the Parquet file metadata
            column_descriptions = get_column_metadata(pq_file)
            column_descriptions = {col: desc.get("description", "") for col, desc in column_descriptions.items() if col in self.columns}
            self.column_descriptions = column_descriptions

    def profile(self) -> 'ParquetProfileReport':
        """
        Profiles the Parquet file

        """
        if self.batch_size is None:
            # Native ydata profiling (no chunking)
            df = pd.read_parquet(self.parquet_path, columns=self.columns)
            self.report = ProfileReport(df, minimal=True, explorative=False, progress_bar=False,
                                        title=self.title, dataset=self.dataset_metadata.to_dict(),
                                        variables=self.column_descriptions)
        else:
            # Columnar profiling
            gen = parquet_column_generator(self.parquet_path, columns=self.columns)
            report = ColumnarProfileReport(
                column_generator=gen,
                column_count=len(self.columns),
                batch_size=self.batch_size,
                show_progress=self.show_progress,
                title=self.title,
                dataset_metadata=self.dataset_metadata,
                column_descriptions=self.column_descriptions)
            report.profile()
            self.report = report.report
        return self

    def to_html(self) -> str:
        """The HTML representation of the profile report."""
        if self.report is None:
            raise RuntimeError("No report generated. Call profile() first.")
        return self.report.to_html()

    def save_html(self, output_html: Path) -> None:
        """ Save the profile report to a HTML file."""
        with atomic_output_file(output_html) as tmp_path:
            tmp_path.write_text(self.to_html(), encoding="utf-8")

    def show(self, notebook: bool = False):
        """Display the profile report in a notebook or open in a browser.

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
