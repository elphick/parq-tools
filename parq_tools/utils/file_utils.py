import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
import os
from typing import ContextManager


@contextmanager
def atomic_output_file(final_file: Path, suffix: str = ".tmp") -> ContextManager[Path]:
    """
    Context manager for atomic file writes using a temporary file.

    All writes are directed to a temporary file in the same directory.
    On successful exit, the temp file is atomically renamed to the final path.
    On error, the temp file is deleted.

    Example:
        .. code-block:: python

            with atomic_output_file(output_path) as tmp_file:
                # Write to tmp_file
                pq.write_table(table, tmp_file)

    Args:
        final_file (Path): The intended final output file path.
        suffix (str): Suffix for the temporary file (default: ".tmp").
    """
    tmp_path = final_file.with_name(final_file.name + suffix)
    try:
        yield tmp_path
        os.replace(tmp_path, final_file)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


@contextmanager
def atomic_output_dir(final_dir: Path, suffix: str = ".tmp") -> ContextManager[Path]:
    """
    Context manager for atomic directory writes using a temporary directory.

    All writes are directed to a temporary directory in the same parent directory.
    On successful exit, the temp directory is atomically renamed to the final directory.
    On error, the temp directory is deleted.

    Example:
        .. code-block:: python

            with atomic_output_dir(final_dir) as tmp_dir:
                # Write files to tmp_dir
                (tmp_dir / "file.txt").write_text("Hello, World!")

    Args:
        final_dir (Path): The intended final output directory path.
        suffix (str): Suffix for the temporary directory (default: ".tmp").

    """
    parent = final_dir.parent
    with tempfile.TemporaryDirectory(dir=parent, suffix=suffix) as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            yield tmp_path
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(tmp_path, final_dir)
        except Exception:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            raise
