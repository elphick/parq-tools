import zipfile
import shutil
from pathlib import Path
import sys
import subprocess
import time

from parq_tools.utils import atomic_output_dir
from parq_tools.utils.optional_imports import get_tqdm


def extract_archive(archive_path: Path,
                    output_dir: Path,
                    show_progress: bool = False) -> None:
    """
    Extracts an archive using `zipfile` or falls back to `7-Zip` if necessary.

    Args:
        archive_path (Path): Path to the archive file.
        output_dir (Path): Directory to extract the contents to.
        show_progress (bool): Whether to display a progress bar. Defaults to False.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tqdm = get_tqdm()

    # Attempt extraction with zipfile
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            file_info = zip_ref.infolist()
            total_size = sum(file.file_size for file in file_info)  # Total size of all files

            if show_progress:
                with tqdm(total=total_size, desc="Extracting", unit="B", unit_scale=True, unit_divisor=1024,
                          dynamic_ncols=True) as pbar:
                    with atomic_output_dir(output_dir) as tmp_dir:
                        for file in file_info:
                            with zip_ref.open(file, 'r') as source, open(tmp_dir / file.filename, 'wb') as target:
                                while chunk := source.read(1024 * 1024):  # Read in chunks
                                    target.write(chunk)
                                    pbar.update(len(chunk))
                                    pbar.refresh()  # Force immediate update
            else:
                zip_ref.extractall(output_dir)
        return
    except (zipfile.BadZipFile, RuntimeError):
        pass  # Fallback to 7-Zip

    # Fallback to 7-Zip
    try:
        extract_archive_with_7zip(archive_path, output_dir, show_progress)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Extraction failed with 7-Zip: {e}")


def extract_archive_with_7zip(archive_path: Path,
                              output_dir: Path,
                              show_progress: bool = False) -> None:
    """
    Extracts an archive using 7-Zip with an optional progress bar.

    Args:
        archive_path (Path): Path to the archive file.
        output_dir (Path): Directory to extract the contents to.
        show_progress (bool): Whether to display a progress bar. Defaults to False.

    """

    tqdm = get_tqdm()

    seven_zip_path = shutil.which("7z")
    if not seven_zip_path:
        raise FileNotFoundError("7-Zip executable not found. Please install 7-Zip and ensure it is in your PATH.")

    # Get the total size of the archive
    total_size = archive_path.stat().st_size

    pbar = None
    if show_progress:
        pbar = tqdm(total=total_size, desc="Extracting", unit="B", unit_scale=True, unit_divisor=1024, file=sys.stderr)

    try:
        with atomic_output_dir(output_dir) as tmp_dir:
            process = subprocess.Popen(
                [seven_zip_path, 'x', str(archive_path), f'-o{tmp_dir}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Monitor the size of extracted files
            while process.poll() is None:
                if pbar:
                    extracted_size = sum(f.stat().st_size for f in tmp_dir.rglob('*') if f.is_file())
                    pbar.n = extracted_size
                    pbar.refresh()
                time.sleep(0.1)  # Avoid excessive CPU usage

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"7-Zip extraction failed with return code {process.returncode}")

            # Ensure progress bar reaches 100% on success
            if pbar:
                pbar.n = total_size
                pbar.refresh()

    finally:
        if pbar:
            pbar.close()
