import os

os.environ["YDATA_SUPPRESS_BANNER"] = "1"

from importlib import metadata
from .parq_concat import ParquetConcat, concat_parquet_files
from .parq_filter import filter_parquet_file
from .parq_schema_tools import rename_and_update_metadata
from .parq_profile import ParquetProfileReport
from .utils.index_utils import reindex_parquet, sort_parquet_file, validate_index_alignment

try:
    __version__ = metadata.version('parq_tools')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
