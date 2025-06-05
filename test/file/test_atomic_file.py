import os
from pathlib import Path
from parq_tools.utils.file_utils import atomic_output_file

def test_atomic_output_file(tmp_path):
    final_file = tmp_path / "result.txt"
    content = "hello atomic file!"
    # Use the context manager to write to the temp file
    with atomic_output_file(final_file) as tmp_file:
        assert not final_file.exists()
        tmp_file.write_text(content)
        assert tmp_file.exists()
    # After context, final file should exist and have the content
    assert final_file.exists()
    assert final_file.read_text() == content

def test_atomic_output_file_cleanup_on_error(tmp_path):
    final_file = tmp_path / "fail.txt"
    try:
        with atomic_output_file(final_file) as tmp_file:
            tmp_file.write_text("fail!")
            raise RuntimeError("fail inside context")
    except RuntimeError:
        pass
    # Temp file and final file should not exist after error
    assert not final_file.exists()
    assert not (tmp_path / "fail.txt.tmp").exists()

