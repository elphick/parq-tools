from importlib import metadata
from .parq_concat import ParquetConcat

try:
    __version__ = metadata.version('parq_tools')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
