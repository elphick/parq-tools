from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def atomic_output_path(final_path: Path, suffix: str = ".tmp"):
    """
    Context manager for atomic file writes using a temporary file.

    All writes are directed to a temporary file in the same directory.
    On successful exit, the temp file is atomically renamed to the final path.
    On error, the temp file is deleted.

    Usage:
        with atomic_output_path(output_path) as tmp_path:
            # Write to tmp_path
            pq.write_table(table, tmp_path)

    Args:
        final_path (Path): The intended final output file path.
        suffix (str): Suffix for the temporary file (default: ".tmp").
    """
    tmp_path = final_path.with_name(final_path.name + suffix)
    try:
        yield tmp_path
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
