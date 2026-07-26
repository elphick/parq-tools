"""
parq_profile.py

Utilities for profiling Parquet files and generating HTML reports using fg-data-profiling, with support for notebook and browser display.

Main API:

- ParquetProfileReport: Class for generating, saving, and displaying profile reports for Parquet files.
"""
import contextlib
from io import StringIO
from dataclasses import dataclass
import copy
import html as html_lib
import json
from pathlib import Path
from typing import Any, Iterator, Optional, List, Mapping, Union, Iterable
import pandas as pd
import pyarrow.parquet as pq


from parq_tools.utils import atomic_output_file
from parq_tools.utils.metadata_utils import get_table_metadata, get_column_metadata, get_pandas_metadata
from parq_tools.utils.profile_utils import (
    ColumnarProfileReport,
    ProfileMetadata,
    ColumnMetadata,
    build_column_descriptions,
)
from parq_tools.utils.optional_imports import (
    get_data_profile_report,
    get_data_profile_compare,
    get_yaml_module,
)
from parq_tools.utils.profile_compare_utils import (
    DEFAULT_COMPARISON_METRICS,
    build_profile_comparison_summary,
    get_changed_columns_from_summary,
    prune_description_to_columns,
)


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
                 column_descriptions: Optional[
                     dict[str, Union[str, Mapping[str, Any], ColumnMetadata]]
                 ] = None) -> None:
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
            column_descriptions: Optional[dict[str, Union[str, Mapping[str, Any], ColumnMetadata]]]:
                Column metadata/description values for the dataset. Supports legacy strings and structured
                metadata payloads. Will over-ride any descriptions in the Parquet file.
        """
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.title = title
        self.report: Optional[object] = None

        metadata = dataset_metadata if isinstance(dataset_metadata, ProfileMetadata) else ProfileMetadata.from_dict(
            dataset_metadata) if dataset_metadata else None
        self.dataset_metadata = metadata
        pq_file = pq.ParquetFile(str(self.parquet_path))
        self.columns = pq_file.schema.names if columns is None else columns
        if not self.dataset_metadata:
            # If no metadata is provided, use the Parquet file metadata
            table_meta: dict = get_table_metadata(pq_file)
            self.dataset_metadata = ProfileMetadata.from_dict(table_meta) if pq_file.metadata else None
        if column_descriptions is None:
            # If no column descriptions are provided, use the Parquet file metadata
            column_descriptions = {
                col: desc for col, desc in get_column_metadata(pq_file).items() if col in self.columns
            }
        else:
            column_descriptions = {
                col: desc for col, desc in column_descriptions.items() if col in self.columns
            }

        self.column_descriptions = build_column_descriptions(column_descriptions)

    def profile(self) -> 'ParquetProfileReport':
        """Profiles the Parquet file."""
        if self.batch_size is None:
            # Native ydata profiling (no chunking)
            ProfileReport = get_data_profile_report("ParquetProfileReport.profile()")
            df = pd.read_parquet(self.parquet_path, columns=self.columns)
            dataset_config = self.dataset_metadata.to_dict() if self.dataset_metadata else {}
            self.report = ProfileReport(df, minimal=True, explorative=False, progress_bar=False,
                                        title=self.title, dataset=dataset_config,
                                        variables={"descriptions": self.column_descriptions})
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

    def to_file(self, output_file: Union[str, Path]) -> None:
        """Save the profile report to an HTML file (data_profiling-compatible name)."""
        self.save_html(Path(output_file))

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


@dataclass
class ParquetProfileComparisonBundle:
    """Reusable outputs from one parquet profile-comparison run."""

    comparison_report: Any
    dataset_descriptions: List[Any]
    labels: List[str]

    @staticmethod
    def _validate_description_status_labels(description_status_labels: str) -> None:
        if description_status_labels not in ("none", "emoji"):
            raise ValueError("description_status_labels must be 'none' or 'emoji'.")

    @staticmethod
    def _apply_description_status_prefixes(
        report: Any,
        summary: dict[str, Any],
        description_status_labels: str,
        prefix_same: str,
        prefix_different: str,
    ) -> Any:
        if description_status_labels == "none":
            return report

        report_copy = copy.deepcopy(report)
        descriptions = getattr(report_copy.config.variables, "descriptions", None)
        if not isinstance(descriptions, dict):
            return report_copy

        report_columns = set(report_copy.get_description().variables.keys())
        for column_name in report_columns:
            current_text = descriptions.get(column_name, "")
            column_summary = summary["columns"].get(column_name, {})
            status = column_summary.get("status")
            prefix = prefix_different if status != "equal" else prefix_same
            if isinstance(current_text, str) and current_text.strip():
                descriptions[column_name] = f"{prefix}{current_text}"
            else:
                descriptions[column_name] = prefix.rstrip().rstrip("|").strip()

        return report_copy

    @staticmethod
    def _build_tolerance_footer_note(
        abs_tol: float,
        rel_tol: float,
        metrics: Optional[Iterable[str]],
    ) -> str:
        metric_list = list(metrics) if metrics is not None else list(DEFAULT_COMPARISON_METRICS)
        metric_text = ", ".join(metric_list)
        return (
            f"Comparison tolerance settings: abs_tol={abs_tol}, rel_tol={rel_tol}, "
            f"metrics=[{metric_text}]"
        )

    @staticmethod
    def _append_footer_note(html: str, note: str) -> str:
        escaped_note = html_lib.escape(note)
        footer_block = (
            "<div style=\"margin:12px 24px 24px 24px;font-size:12px;color:#666;\">"
            f"{escaped_note}</div>"
        )
        if "</body>" in html:
            return html.replace("</body>", f"{footer_block}</body>", 1)
        return f"{html}{footer_block}"

    @staticmethod
    def _format_comparison_title(labels: List[str]) -> str:
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"Comparing {labels[0]} and {labels[1]}"
        return f"Comparing {', '.join(labels[:-1])} and {labels[-1]}"

    @classmethod
    def _write_report_html_with_footer(
        cls,
        report: Any,
        output_path: Path,
        footer_note: str,
    ) -> None:
        html = report.to_html()
        html = cls._append_footer_note(html, footer_note)
        with atomic_output_file(output_path) as tmp_path:
            tmp_path.write_text(html, encoding="utf-8")

    def to_summary_dict(
        self,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        metrics: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        return build_profile_comparison_summary(
            descriptions=self.dataset_descriptions,
            labels=self.labels,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            metrics=metrics,
        )

    def to_summary_json(
        self,
        path: Optional[Union[str, Path]] = None,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        metrics: Optional[Iterable[str]] = None,
        indent: int = 2,
    ) -> str:
        summary = self.to_summary_dict(abs_tol=abs_tol, rel_tol=rel_tol, metrics=metrics)
        text = json.dumps(summary, indent=indent, sort_keys=True)
        if path is not None:
            output_path = Path(path)
            with atomic_output_file(output_path) as tmp_path:
                tmp_path.write_text(text, encoding="utf-8")
        return text

    def to_summary_yaml(
        self,
        path: Optional[Union[str, Path]] = None,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        metrics: Optional[Iterable[str]] = None,
    ) -> str:
        yaml = get_yaml_module("ParquetProfileComparisonBundle.to_summary_yaml()")
        summary = self.to_summary_dict(abs_tol=abs_tol, rel_tol=rel_tol, metrics=metrics)
        text = yaml.safe_dump(summary, sort_keys=False)
        if path is not None:
            output_path = Path(path)
            with atomic_output_file(output_path) as tmp_path:
                tmp_path.write_text(text, encoding="utf-8")
        return text

    def to_diff_report(
        self,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        metrics: Optional[Iterable[str]] = None,
        description_status_labels: str = "none",
        prefix_same: str = "🟢 SAME | ",
        prefix_different: str = "🔴 DIFF | ",
    ):
        self._validate_description_status_labels(description_status_labels)
        compare_reports = get_data_profile_compare("ParquetProfileComparisonBundle.to_diff_report()")
        summary = self.to_summary_dict(abs_tol=abs_tol, rel_tol=rel_tol, metrics=metrics)
        changed_columns = get_changed_columns_from_summary(summary)
        pruned_descriptions = [
            prune_description_to_columns(desc, changed_columns) for desc in self.dataset_descriptions
        ]
        report = compare_reports(pruned_descriptions)
        if not changed_columns:
            comparison_title = self._format_comparison_title(self.labels)
            report.config.title = comparison_title
            report._description_set.analysis.title = comparison_title
        return self._apply_description_status_prefixes(
            report=report,
            summary=summary,
            description_status_labels=description_status_labels,
            prefix_same=prefix_same,
            prefix_different=prefix_different,
        )

    def to_comparison_report(
        self,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        metrics: Optional[Iterable[str]] = None,
        description_status_labels: str = "none",
        prefix_same: str = "🟢 SAME | ",
        prefix_different: str = "🔴 DIFF | ",
    ):
        self._validate_description_status_labels(description_status_labels)
        summary = self.to_summary_dict(abs_tol=abs_tol, rel_tol=rel_tol, metrics=metrics)
        return self._apply_description_status_prefixes(
            report=self.comparison_report,
            summary=summary,
            description_status_labels=description_status_labels,
            prefix_same=prefix_same,
            prefix_different=prefix_different,
        )

    def write_outputs(
        self,
        comparison_html: Optional[Union[str, Path]] = None,
        diff_html: Optional[Union[str, Path]] = None,
        differences_yaml: Optional[Union[str, Path]] = None,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        metrics: Optional[Iterable[str]] = None,
        description_status_labels: str = "none",
        prefix_same: str = "🟢 SAME | ",
        prefix_different: str = "🔴 DIFF | ",
    ) -> dict[str, Path]:
        self._validate_description_status_labels(description_status_labels)
        footer_note = self._build_tolerance_footer_note(
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            metrics=metrics,
        )
        written: dict[str, Path] = {}
        if comparison_html is not None:
            comparison_path = Path(comparison_html)
            comparison_report = self.to_comparison_report(
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                metrics=metrics,
                description_status_labels=description_status_labels,
                prefix_same=prefix_same,
                prefix_different=prefix_different,
            )
            self._write_report_html_with_footer(comparison_report, comparison_path, footer_note)
            written["comparison_html"] = comparison_path
        if diff_html is not None:
            diff_path = Path(diff_html)
            diff_report = self.to_diff_report(
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                metrics=metrics,
                description_status_labels=description_status_labels,
                prefix_same=prefix_same,
                prefix_different=prefix_different,
            )
            self._write_report_html_with_footer(diff_report, diff_path, footer_note)
            written["diff_html"] = diff_path
        if differences_yaml is not None:
            yaml_path = Path(differences_yaml)
            self.to_summary_yaml(path=yaml_path, abs_tol=abs_tol, rel_tol=rel_tol, metrics=metrics)
            written["differences_yaml"] = yaml_path
        return written


def build_parquet_profile_comparison(
    parquet_paths: List[Union[str, Path]],
    columns: Optional[List[str]] = None,
    batch_size: Optional[int] = 1,
    show_progress: bool = True,
    titles: Optional[List[str]] = None,
    dataset_metadata: Optional[List[Optional[Union[dict, ProfileMetadata]]]] = None,
    column_descriptions: Optional[dict[str, Union[str, Mapping[str, Any], ColumnMetadata]]] = None,
) -> ParquetProfileComparisonBundle:
    """Compare 2 or 3 parquet files using profiling reports.

    Uses memory-managed columnar profiling when ``batch_size`` is an integer.
    """
    file_count = len(parquet_paths)
    if file_count not in (2, 3):
        raise ValueError("parquet_paths must contain exactly 2 or 3 file paths.")

    if titles is not None and len(titles) != file_count:
        raise ValueError("titles must have the same length as parquet_paths.")

    if dataset_metadata is not None and len(dataset_metadata) != file_count:
        raise ValueError("dataset_metadata must have the same length as parquet_paths.")

    metadata_list = dataset_metadata if dataset_metadata is not None else [None] * file_count
    compare_reports = get_data_profile_compare("compare_parquet_profiles()")

    reports = []
    labels = []
    for idx, parquet_path in enumerate(parquet_paths):
        report_title = titles[idx] if titles is not None else f"Dataset {chr(65 + idx)}"
        labels.append(report_title)
        profiler = ParquetProfileReport(
            parquet_path=parquet_path,
            columns=columns,
            batch_size=batch_size,
            show_progress=show_progress,
            title=report_title,
            dataset_metadata=metadata_list[idx],
            column_descriptions=column_descriptions,
        )
        profiler.profile()
        if profiler.report is None:
            raise RuntimeError(f"No report generated for {parquet_path}.")
        reports.append(profiler.report.get_description())

    comparison_report = compare_reports(reports)
    return ParquetProfileComparisonBundle(
        comparison_report=comparison_report,
        dataset_descriptions=reports,
        labels=labels,
    )


def compare_parquet_profiles(
    parquet_paths: List[Union[str, Path]],
    columns: Optional[List[str]] = None,
    batch_size: Optional[int] = 1,
    show_progress: bool = True,
    titles: Optional[List[str]] = None,
    dataset_metadata: Optional[List[Optional[Union[dict, ProfileMetadata]]]] = None,
    column_descriptions: Optional[dict[str, Union[str, Mapping[str, Any], ColumnMetadata]]] = None,
):
    """Return only the merged comparison report for backward compatibility."""
    bundle = build_parquet_profile_comparison(
        parquet_paths=parquet_paths,
        columns=columns,
        batch_size=batch_size,
        show_progress=show_progress,
        titles=titles,
        dataset_metadata=dataset_metadata,
        column_descriptions=column_descriptions,
    )
    return bundle.comparison_report
