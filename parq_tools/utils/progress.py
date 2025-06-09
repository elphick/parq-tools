import math
from typing import Sequence, Optional
from parq_tools.utils.optional_imports import get_tqdm

def get_batch_progress_bar(
    datasets: Sequence[object],  # Assuming datasets have a count_rows() method
    batch_size: int,
    desc: Optional[str] = None,
    disable: bool = False
) -> 'tqdm':
    """
    Utility to create a tqdm progress bar for batch processing of pyarrow datasets.

    Args:
        datasets (Sequence): Sequence of datasets (must have .count_rows()).
        batch_size (int): Number of rows per batch.
        desc (str, optional): Description for the progress bar.
        disable (bool, optional): If True, disables the progress bar.

    Returns:
        tqdm: A tqdm progress bar instance (or dummy if tqdm is not installed).
    """
    tqdm = get_tqdm()
    total_rows = sum(dataset.count_rows() for dataset in datasets)
    total_batches = max(math.ceil(total_rows / batch_size), 1)
    return tqdm(total=total_batches, desc=desc, disable=disable)

