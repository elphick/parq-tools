from pathlib import Path
from typing import Iterator, Optional, Union
import pandas as pd
import os

from ydata_profiling import ProfileReport

from parq_tools.utils import atomic_output_file

try:
    # noinspection PyUnresolvedReferences
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ColumnarProfileReport:
    """Memory-efficient, column-wise profiler for large datasets using ydata-profiling.

    This class can be leveraged by any file reader that can yield pandas Series.
    """

    def __init__(self,
                 column_generator: Iterator[pd.Series],
                 column_count: Optional[int] = None,
                 batch_size: int = 1,
                 show_progress: bool = True):
        """
        Initialize the ColumnarProfileReport.
        This profiler processes columns in batches, allowing for profiling large datasets without loading them
        entirely into memory.

        Args:
            column_generator: A generator or iterable that yields pandas Series.
            column_count: The total number of columns used by the progressbar.
            batch_size: The number of columns to process in each batch.
            show_progress: If True, displays a progress bar during profiling.
        """


        self.column_generator = column_generator
        self.column_count = column_count
        self.batch_size = batch_size
        self.show_progress = show_progress and HAS_TQDM
        self.head_report: ProfileReport | None = None
        self.report: ProfileReport | None = None
        self.index_memory: int = 0

    def profile(self) -> None:
        col_names = []
        descriptions = []
        head_chunks: list[pd.DataFrame] = []

        total_columns = self.column_count

        from itertools import islice

        def batched(iterable, batch_size):
            it = iter(iterable)
            while True:
                batch = list(islice(it, batch_size))
                if not batch:
                    break
                yield batch

        total_progress_steps = total_columns + 1 if total_columns else None
        progress = tqdm(total=total_progress_steps, desc="Profiling columns",
                        leave=True) if self.show_progress else None

        for batch in batched(self.column_generator, self.batch_size):
            batch_names = []
            for col in batch:
                if self.index_memory == 0:
                    self.index_memory = col.index.memory_usage(deep=True) if hasattr(col, 'index') else 0
                if hasattr(col, "name") and col.name is not None:
                    batch_names.append(str(col.name))
                else:
                    batch_names.append(f"col_{len(col_names) + len(batch_names)}")
            df = pd.DataFrame({name: col for name, col in zip(batch_names, batch)})
            head_chunks.append(df.head())
            report = ProfileReport(df, minimal=True, explorative=False, progress_bar=False)
            # descriptions.append(report.get_description())  # issue with unmanage progress bar
            desc = BatchDescription(report.config, df, report.summarizer, report.typeset)
            descriptions.append(desc)

            col_names.extend(batch_names)
            if progress:
                progress.update(len(batch))

        if not head_chunks:
            raise ValueError("No columns were provided to profile.")

        # profile the head chunks
        head_df = pd.concat(head_chunks, axis=1)
        head_report = ProfileReport(head_df, minimal=True, explorative=False, progress_bar=False)
        if progress:
            progress.update(1)
            progress.close()

        self.head_report = head_report

        self.report = self._combine_reports(descriptions)

    def _combine_reports(self, descriptions):
        import copy
        final_report = copy.deepcopy(self.head_report)

        # Merge variable summaries
        for desc in descriptions:
            for var, var_summary in desc.variables.items():
                final_report.description_set.variables[var] = var_summary

        # Recalculate overview
        overview = final_report.description_set.table
        n = descriptions[0].table.get("n", 0)
        overview["n"] = n

        # Get total memory by summing per-column memory (each includes index)
        total_column_memory = sum(desc.table["memory_size"] for desc in descriptions)
        # Subtract index memory (n-1) times
        n = len(descriptions)
        total_memory = total_column_memory - self.index_memory * (n - 1)
        overview["memory_size"] = total_memory
        overview["record_size"] = total_memory / overview["n"] if overview["n"] else 0

        # Merge alerts
        all_alerts = []
        for desc in descriptions:
            all_alerts.extend(desc.alerts)
        final_report.description_set.alerts = all_alerts

        final_report.df = self.head_report.df  # or None
        return final_report

    @staticmethod
    def _merge_overview(target, source):
        for k, v in source.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                ColumnarProfileReport._merge_overview(target[k], v)
            elif isinstance(v, (int, float)) and isinstance(target.get(k), (int, float)):
                target[k] = target.get(k, 0) + v
            elif isinstance(v, list) and isinstance(target.get(k), list):
                target[k] = target.get(k, []) + v
            elif isinstance(v, set) and isinstance(target.get(k), set):
                target[k] = target.get(k, set()).union(v)
            else:
                target[k] = v

    def to_html(self) -> str:
        if self.report is None:
            raise RuntimeError("No report generated. Call profile() first.")
        return self.report.to_html()

    def save_html(self, output_html: Path) -> None:
        with atomic_output_file(output_html) as tmp_path:
            tmp_path.write_text(self.to_html(), encoding="utf-8")

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


class BatchDescription:
    """A class to patch ydata-profiling progressbar bug

    As at ydata-profiling=4.16.1 there is a bug with the progress bar that does not respect the
    `progress_bar` parameter in the `ProfileReport` constructor. This class is used to create a
    description of a batch of columns, mimicking the behavior of `ydata_profiling.model.pandas.describe_1d`

    TODO: report the ydata-profiling unmanaged progressbar bug for an upstream fix

    """
    def __init__(self, config, df, summarizer, typeset):
        from ydata_profiling.model.pandas.summary_pandas import pandas_describe_1d
        from ydata_profiling.model.table import get_table_stats
        from ydata_profiling.model.alerts import get_alerts

        self.variables = {
            name: pandas_describe_1d(config, series, summarizer, typeset)
            for name, series in df.items()
        }
        self.table = get_table_stats(config, df, self.variables)
        self.alerts = get_alerts(config, self.table, self.variables, correlations={})
