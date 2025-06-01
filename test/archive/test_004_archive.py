import pytest
from pathlib import Path
import tempfile
import zipfile
from parq_tools.utils.archive_utils import extract_archive, extract_archive_with_7zip

import os
import zipfile
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def large_zip_file():
    """
    Fixture to create a large zip file for testing, cached across test sessions.
    """
    temp_dir = Path(tempfile.gettempdir())
    archive_path = temp_dir / "large_test.zip"
    num_files = 5
    file_size = 1024 * 1024 * 1024  # 1 GB per file

    if not archive_path.exists():
        temp_files_dir = temp_dir / "temp_files"
        temp_files_dir.mkdir(parents=True, exist_ok=True)

        # Generate large files
        for i in range(num_files):
            file_path = temp_files_dir / f"file_{i}.txt"
            with open(file_path, "wb") as f:
                f.write(os.urandom(file_size))

        # Create the zip file
        with zipfile.ZipFile(archive_path, "w") as zipf:
            for file in temp_files_dir.iterdir():
                zipf.write(file, arcname=file.name)

        # Clean up temporary files
        for file in temp_files_dir.iterdir():
            file.unlink()
        temp_files_dir.rmdir()

    return archive_path


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp:
        yield Path(temp)


def create_test_zip(archive_path, file_name, content):
    with zipfile.ZipFile(archive_path, 'w') as zipf:
        zipf.writestr(file_name, content)


def test_extract_single_file(temp_dir):
    archive_path = temp_dir / "test.zip"
    output_dir = temp_dir / "output"
    create_test_zip(archive_path, "test.txt", "This is a test file.")

    extract_archive(archive_path, output_dir, show_progress=False)

    extracted_file = output_dir / "test.txt"
    assert extracted_file.exists()
    assert extracted_file.read_text() == "This is a test file."


@pytest.mark.parametrize("show_progress", [False, True])
def test_extract_with_progress(temp_dir, large_zip_file, show_progress, capsys):
    with capsys.disabled():  # to allow the progress bar to display correctly

        output_dir = temp_dir / "output"
        extract_archive(large_zip_file, output_dir, show_progress=show_progress)

        # inspect the files inside the archive
        with zipfile.ZipFile(large_zip_file) as zipf:
            file_list = zipf.namelist()
            assert len(file_list) > 0, "The archive should contain files."

        # check the files exist in the output directory
        for file in file_list:
            extracted_file = output_dir / file
            assert extracted_file.exists(), f"File {file} should be extracted."
            assert extracted_file.stat().st_size > 0, f"File {file} should not be empty."


def test_invalid_zip(temp_dir):
    archive_path = temp_dir / "invalid.zip"
    output_dir = temp_dir / "output"
    with open(archive_path, 'w') as f:
        f.write("Not a valid zip file")

    with pytest.raises(RuntimeError):
        extract_archive(archive_path, output_dir, show_progress=False)


def test_extract_with_7zip(temp_dir):
    # This test assumes that 7-Zip is installed and available in the system PATH.
    archive_path = temp_dir / "test.7z"
    output_dir = temp_dir / "output"
    create_test_zip(archive_path, "test.txt", "This is a test file.")

    # Simulate a 7-Zip extraction by renaming the file to .7z
    archive_path.rename(temp_dir / "test.7z")

    extract_archive_with_7zip(temp_dir / "test.7z", output_dir, show_progress=False)

    extracted_file = output_dir / "test.txt"
    assert extracted_file.exists()
    assert extracted_file.read_text() == "This is a test file."


@pytest.mark.parametrize("show_progress", [True])
def test_extract_with_7zip_with_progress(temp_dir, large_zip_file, show_progress, capsys):
    with capsys.disabled():  # to allow the progress bar to display correctly

        # This test assumes that 7-Zip is installed and available in the system PATH.
        output_dir = temp_dir / "output"

        # inspect the files inside the archive
        with zipfile.ZipFile(large_zip_file) as zipf:
            file_list = zipf.namelist()
            assert len(file_list) > 0, "The archive should contain files."

        extract_archive_with_7zip(large_zip_file, output_dir, show_progress=show_progress)

        # check the files exist in the output directory
        for file in file_list:
            extracted_file = output_dir / file
            assert extracted_file.exists(), f"File {file} should be extracted."
            assert extracted_file.stat().st_size > 0, f"File {file} should not be empty."
