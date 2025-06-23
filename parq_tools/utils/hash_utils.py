import hashlib
from pathlib import Path


def file_hash(path: Path, hash_func=hashlib.sha256, chunk_size=1024 * 1024, show_progress=False) -> str:
    total = path.stat().st_size
    h = hash_func()
    try:
        from tqdm import tqdm
        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False
    with open(path, "rb") as f:
        if use_tqdm:
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"Hashing {path.name}") as pbar:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    h.update(chunk)
                    pbar.update(len(chunk))
        else:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
    return h.hexdigest()


def fast_file_check(src: Path, dst: Path, sample_size=4096):
    if not dst.exists():
        return False
    if src.stat().st_size != dst.stat().st_size:
        return False
    if int(src.stat().st_mtime) != int(dst.stat().st_mtime):
        return False
    size = src.stat().st_size
    with open(src, "rb") as fsrc, open(dst, "rb") as fdst:
        # Check first, middle, and last sample_size bytes
        for offset in [0, max(0, size // 2 - sample_size // 2), max(0, size - sample_size)]:
            fsrc.seek(offset)
            fdst.seek(offset)
            if fsrc.read(sample_size) != fdst.read(sample_size):
                return False
    return True


def _select_hash_func(hash_method):
    if hash_method == "sha256":
        return hashlib.sha256
    elif hash_method == "xxhash":
        try:
            import xxhash
        except ImportError:
            raise RuntimeError("xxhash is not installed")
        return xxhash.xxh64
    elif callable(hash_method):
        return hash_method
    else:
        raise ValueError(f"Unknown hash_method: {hash_method}")


def files_match(src: Path, dst: Path, hash_method='fast', chunk_size: int = 1024 * 1024, show_progress: bool = False) -> bool:
    if not src.exists():
        raise FileNotFoundError(f"Source file {src} does not exist.")
    if not dst.exists():
        return False

    if hash_method == "fast":
        return fast_file_check(src, dst)
    h = _select_hash_func(hash_method)
    src_hash = file_hash(src, hash_func=h, chunk_size=chunk_size, show_progress=show_progress)
    dst_hash = file_hash(dst, hash_func=h, chunk_size=chunk_size, show_progress=show_progress)
    return dst.exists() and src_hash == dst_hash
