import time

import pytest
from pathlib import Path
from parq_tools.utils.file_utils import atomic_file_copy


def create_file(path: Path, content: bytes):
    path.write_bytes(content)


def test_basic_copy(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    create_file(src, b"hello world")
    dst = atomic_file_copy(src, dst)
    assert dst.read_bytes() == b"hello world"


def test_skip_if_files_match(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    create_file(src, b"abc")
    create_file(dst, b"abc")
    mtime_before = dst.stat().st_mtime
    atomic_file_copy(src, dst)
    assert dst.read_bytes() == b"abc"
    assert dst.stat().st_mtime == mtime_before  # Not overwritten


def test_force_copy(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    create_file(src, b"abc")
    create_file(dst, b"abc")
    mtime_before = dst.stat().st_mtime
    time.sleep(1)  # Ensure mtime changes
    atomic_file_copy(src, dst, force=True)
    assert dst.read_bytes() == b"abc"
    assert dst.stat().st_mtime != mtime_before  # Overwritten


def test_copy_to_directory(tmp_path):
    src = tmp_path / "src.txt"
    outdir = tmp_path / "out"
    outdir.mkdir()
    create_file(src, b"xyz")
    result = atomic_file_copy(src, outdir)
    assert (outdir / "src.txt").read_bytes() == b"xyz"
    assert result == outdir / "src.txt"


def test_nonexistent_source(tmp_path):
    src = tmp_path / "nope.txt"
    dst = tmp_path / "dst.txt"
    with pytest.raises(FileNotFoundError):
        atomic_file_copy(src, dst)

def test_copy_with_custom_hash(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    create_file(src, b"custom hash test")

    # Using a custom hash function (e.g., hashlib.md5)
    import hashlib
    def custom_hash():
        return hashlib.md5()

    atomic_file_copy(src, dst, hash_method=custom_hash)
    assert dst.read_bytes() == b"custom hash test"

def test_copy_large_file_with_progress(tmp_path):
    src = tmp_path / "large_src.txt"
    dst = tmp_path / "large_dst.txt"
    # Create a large file
    create_file(src, b"x" * (10 * 1024 * 1024))  # 10 MB file
    result = atomic_file_copy(src, dst, show_progress=True)
    assert result == dst
    assert dst.stat().st_size == src.stat().st_size
    assert dst.read_bytes() == b"x" * (10 * 1024 * 1024)

def test_copy_with_chunk_size(tmp_path):
    src = tmp_path / "chunk_src.txt"
    dst = tmp_path / "chunk_dst.txt"
    create_file(src, b"chunked copy test")

    # Copy with a specific chunk size
    atomic_file_copy(src, dst, chunk_size=5)
    assert dst.read_bytes() == b"chunked copy test"
    assert dst.stat().st_size == len(b"chunked copy test")

def test_copy_with_invalid_hash_method(tmp_path):
    src = tmp_path / "invalid_hash_src.txt"
    dst = tmp_path / "invalid_hash_dst.txt"
    create_file(src, b"test for invalid hash method")

    # Using an invalid hash method
    with pytest.raises(ValueError):
        atomic_file_copy(src, dst, hash_method="invalid_hash")

def test_copy_with_callable_hash(tmp_path):
    src = tmp_path / "callable_hash_src.txt"
    dst = tmp_path / "callable_hash_dst.txt"
    create_file(src, b"test for callable hash method")

    # Using a callable hash function
    def my_hash():
        import hashlib
        return hashlib.sha1()

    atomic_file_copy(src, dst, hash_method=my_hash)
    assert dst.read_bytes() == b"test for callable hash method"
    assert dst.stat().st_size == len(b"test for callable hash method")