import logging
import pyarrow.dataset as ds
import pyarrow as pa
from typing import List


def validate_index_alignment(datasets: List[ds.Dataset], index_columns: List[str], batch_size: int = 1024) -> None:
    """
    Validates that the index columns are identical across all datasets.

    Args:
        datasets (List[ds.Dataset]): List of PyArrow datasets to validate.
        index_columns (List[str]): List of index column names to compare.
        batch_size (int, optional): Number of rows per batch to process. Defaults to 1024.

    Raises:
        ValueError: If the index columns are not identical across datasets.
    """
    logging.info("Validating index alignment across datasets")
    scanners = [dataset.scanner(columns=index_columns, batch_size=batch_size) for dataset in datasets]
    iterators = [scanner.to_batches() for scanner in scanners]

    reference_batch = None

    while True:
        current_batches = []
        all_exhausted = True

        for iterator in iterators:
            try:
                batch = next(iterator)
                current_batches.append(pa.Table.from_batches([batch]))
                all_exhausted = False
            except StopIteration:
                current_batches.append(None)

        if all_exhausted:
            break

        reference_batch = current_batches[0]
        for i, current_batch in enumerate(current_batches[1:], start=1):
            if current_batch is not None and not current_batch.equals(reference_batch):
                raise ValueError(
                    f"Index columns are not aligned across datasets. Mismatch found in dataset {i}."
                )

    logging.info("Index alignment validated successfully")