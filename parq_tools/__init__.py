from importlib import metadata
from .parq_concat import ParquetConcat, concat_parquet_files
from .parq_filter import filter_parquet_file
from .utils.index_utils import reindex_parquet, sort_parquet_file, validate_index_alignment


try:
    __version__ = metadata.version('parq_tools')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
