import pyarrow.parquet as pq
import hashlib


def hash_recordbatch(batch):
    # Hash the concatenated bytes of all columns in the RecordBatch
    m = hashlib.sha256()
    for array in batch.columns:
        for buf in array.buffers():
            if buf is not None:
                m.update(buf)
    return m.hexdigest()


def compare_metadata(meta1, meta2):
    # Compare file metadata
    return meta1 == meta2

def compare_parquet_files(file1, file2, chunk_size=10000, show_progress=False):
    result = {
        'metadata': False,
        'columns': {},
        'columns_match': False,
        'missing_columns': {'left_only': [], 'right_only': []},
        'num_rows_match': False,
        'num_rows_left': 0,
        'num_rows_right': 0,
    }
    pf1 = pq.ParquetFile(file1)
    pf2 = pq.ParquetFile(file2)

    # Compare metadata
    result['metadata'] = compare_metadata(pf1.metadata.metadata, pf2.metadata.metadata)

    # Compare columns
    cols1 = set(pf1.schema.names)
    cols2 = set(pf2.schema.names)
    result['columns_match'] = cols1 == cols2
    result['missing_columns']['left_only'] = sorted(list(cols1 - cols2))
    result['missing_columns']['right_only'] = sorted(list(cols2 - cols1))
    all_cols = cols1 | cols2

    # Compare column dtypes
    result['dtypes'] = {}
    for col in all_cols:
        left_dtype = str(pf1.schema_arrow.field(col).type) if col in cols1 else None
        right_dtype = str(pf2.schema_arrow.field(col).type) if col in cols2 else None
        match = (left_dtype == right_dtype) and (left_dtype is not None)
        result['dtypes'][col] = {'left': left_dtype, 'right': right_dtype, 'match': match}

    # Compare row counts
    result['num_rows_left'] = pf1.metadata.num_rows
    result['num_rows_right'] = pf2.metadata.num_rows
    result['num_rows_match'] = pf1.metadata.num_rows == pf2.metadata.num_rows


    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    if show_progress and tqdm is not None:
        col_iter = tqdm(sorted(all_cols), desc="Comparing columns")
    else:
        col_iter = sorted(all_cols)

    for col in col_iter:
        if col not in cols1 or col not in cols2:
            result['columns'][col] = False
            continue

        col_equal = True
        for batch1, batch2 in zip(
                pf1.iter_batches(columns=[col], batch_size=chunk_size),
                pf2.iter_batches(columns=[col], batch_size=chunk_size)
        ):
            if hash_recordbatch(batch1) != hash_recordbatch(batch2):
                col_equal = False
                break
        result['columns'][col] = col_equal

    return result