import inspect
import json
from pathlib import Path

import pyarrow.parquet as pq


class CalculatedColumn:
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.dependencies = list(inspect.signature(func).parameters)

    def evaluate(self, data):
        args = [data[dep] for dep in self.dependencies]
        return self.func(*args)


class CalculatedParquetReader:
    def __init__(self, parquet_path: Path, columns: list[str], calculated_columns: list[CalculatedColumn]):
        self.parquet_path = parquet_path
        self.base_columns = columns
        self.calculated_columns = calculated_columns
        self.columns = self._get_column_order()

    def _get_column_order(self):
        # Return columns in order, placing calculated columns after their last dependency
        order = list(self.base_columns)
        for col in self.calculated_columns:
            last_dep = max(order.index(dep) for dep in col.dependencies)
            order.insert(last_dep + 1, col.name)
        return order

    def read(self, columns=None):
        import pyarrow.parquet as pq
        if columns is None:
            columns = self.base_columns
        else:
            columns = [col for col in columns if col in self.base_columns]
        table = pq.read_table(self.parquet_path, columns=columns)
        df = table.to_pandas()
        for col in self.calculated_columns:
            df[col.name] = col.evaluate(df)
        df = df[self.columns]
        return df

    def iter_chunks(self, chunk_size):
        for batch in pq.ParquetFile(self.parquet_path).iter_batches(batch_size=chunk_size,
                                                                    columns=self.base_columns):
            df = batch.to_pandas()
            for col in self.calculated_columns:
                df[col.name] = col.evaluate(df)
            df = df[self.columns]
            yield df

    def save(self, parquet_path: Path):
        meta = {
            "calculated_columns": [
                {"name": col.name, "func": col.func.__code__.co_code.hex()}
                for col in self.calculated_columns
            ]
        }
        # Write metadata (this is a simplified example)
        pq.write_metadata(self.parquet_path, json.dumps(meta))

    @classmethod
    def load(cls, parquet_path):
        # Load calculated column info from Parquet metadata
        meta = pq.read_metadata(parquet_path)
        meta_dict = json.loads(meta)
        calculated_columns = []
        for col in meta_dict.get("calculated_columns", []):
            # WARNING: In practice, reconstructing functions from code is unsafe and non-trivial.
            # Here, you would use a safe deserialization method.
            pass
        # You would also extract base columns from the Parquet file
        return cls(parquet_path, columns=[], calculated_columns=calculated_columns)
