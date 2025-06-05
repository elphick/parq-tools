import os
from pathlib import Path
from parq_tools.utils.file_utils import atomic_output_dir

def test_atomic_output_dir(tmp_path):
    final_dir = tmp_path / "result_dir"
    file_name = "file.txt"
    content = "atomic dir!"
    # Use the context manager to write to the temp dir
    with atomic_output_dir(final_dir) as tmp_dir:
        assert not final_dir.exists()
        file_path = tmp_dir / file_name
        file_path.write_text(content)
        assert file_path.exists()
    # After context, final dir and file should exist
    assert final_dir.exists()
    assert (final_dir / file_name).exists()
    assert (final_dir / file_name).read_text() == content

def test_atomic_output_dir_cleanup_on_error(tmp_path):
    final_dir = tmp_path / "fail_dir"
    file_name = "fail.txt"
    try:
        with atomic_output_dir(final_dir) as tmp_dir:
            (tmp_dir / file_name).write_text("fail!")
            raise RuntimeError("fail inside context")
    except RuntimeError:
        pass
    # Temp dir and final dir should not exist after error
    assert not final_dir.exists()
    # There may be a leftover temp dir if context manager failed before cleanup, but it should be cleaned up by tempfile

