import hashlib
import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
import os
from typing import ContextManager, Union, Callable, Any

from parq_tools.utils.hash_utils import fast_file_check, file_hash, files_match

logger = logging.getLogger(__name__)


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


def atomic_file_copy(src: Path,
                     dst: Path,
                     chunk_size=1024 * 1024,
                     hash_method: Union[str, Callable[[], Any]] = 'sha256',
                     show_progress: bool = False,
                     force: bool = False
                     ) -> Path:
    """
    Copy a file atomically from `src` to `dst`.

    Args:
        src: Path to the source file.
        dst: Path to the destination file or directory.
        chunk_size: Size of chunks to read/write.
        hash_method: One of 'fast', 'sha256', 'xxhash' or a provided hash function.
        show_progress: Show progress bar if True.
        force: If True, copy even if files match.

    Returns:
        Path to the copied file.
    """
    src = Path(src)
    dst = Path(dst)
    if dst.is_dir():
        dst = dst / src.name

    if not force and files_match(src, dst, hash_method=hash_method, chunk_size=chunk_size, show_progress=show_progress):
        logger.debug(f"File {dst} already exists and is identical, skipping.")
        return dst

    total = src.stat().st_size
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with atomic_output_file(dst) as tmp_dst:
        with open(src, "rb") as fsrc, open(tmp_dst, "wb") as fdst:
            if use_tqdm and show_progress:
                with tqdm(total=total, unit="B", unit_scale=True, desc=f"Copying {src.name}") as pbar:
                    for chunk in iter(lambda: fsrc.read(chunk_size), b""):
                        fdst.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in iter(lambda: fsrc.read(chunk_size), b""):
                    fdst.write(chunk)
            shutil.copystat(src, tmp_dst)

    if not files_match(src, dst, hash_method=hash_method, chunk_size=chunk_size, show_progress=show_progress):
        logger.error(f"{hash_method} mismatch after copy: {src} -> {dst}")
        raise RuntimeError(f"{hash_method} mismatch after copy: {src} -> {dst}")
    return dst
