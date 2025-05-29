from pathlib import Path

import pytest
import pandas as pd


@pytest.fixture(scope="session")
def parquet_test_dir(tmp_path_factory):
    # Create a shared temporary directory for all Parquet files
    return Path(tmp_path_factory.mktemp("parquet_test_files"))


@pytest.fixture(scope="session")
def parquet_test_file_1(parquet_test_dir):
    # Define the dataset
    data = {
        "x": range(1, 11),  # Index column
        "y": range(11, 21),  # Index column
        "z": range(21, 31),  # Index column
        "a": [f"val{i}" for i in range(1, 11)],  # Supplementary column
        "b": [i * 2 for i in range(1, 11)],  # Supplementary column
        "c": [i % 3 for i in range(1, 11)],  # Supplementary column
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a Parquet file with 5 rows per chunk
    file_path = parquet_test_dir / "test_data_1.parquet"
    df.to_parquet(file_path, engine="pyarrow", row_group_size=5, index=False)

    # Return the path to the Parquet file
    return file_path


@pytest.fixture(scope="session")
def parquet_test_file_2(parquet_test_dir):
    # Define the dataset
    data = {
        "x": range(1, 11),  # Index column
        "y": range(11, 21),  # Index column
        "z": range(21, 31),  # Index column
        "d": [f"data{i}" for i in range(1, 11)],  # Supplementary column
        "e": [i * 3 for i in range(1, 11)],  # Supplementary column
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a Parquet file
    file_path = parquet_test_dir / "test_data_2.parquet"
    df.to_parquet(file_path, engine="pyarrow", row_group_size=5, index=False)

    # Return the path to the Parquet file
    return file_path


@pytest.fixture(scope="session")
def parquet_test_file_3(parquet_test_dir):
    # Define the dataset
    data = {
        "x": range(1, 11),  # Index column
        "y": range(11, 21),  # Index column
        "z": range(21, 31),  # Index column
        "f": [f"extra{i}" for i in range(1, 11)],  # Supplementary column
        "g": [i ** 2 for i in range(1, 11)],  # Supplementary column
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a Parquet file
    file_path = parquet_test_dir / "test_data_3.parquet"
    df.to_parquet(file_path, engine="pyarrow", row_group_size=5, index=False)

    # Return the path to the Parquet file
    return file_path


@pytest.fixture(scope="session")
def parquet_tall_file_11(parquet_test_dir):
    # Define the dataset
    data = {
        "x": range(1, 11),  # Index column
        "y": range(21, 31),  # Index column
        "z": range(31, 41),  # Index column
        "a": [f"val{i}" for i in range(1, 11)],  # Supplementary column
        "b": [i * 2 for i in range(1, 11)],  # Supplementary column
        "c": [i % 3 for i in range(1, 11)],  # Supplementary column
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a Parquet file
    file_path = parquet_test_dir / "test_data_11.parquet"
    df.to_parquet(file_path, engine="pyarrow", row_group_size=5, index=False)

    # Return the path to the Parquet file
    return file_path


@pytest.fixture(scope="session")
def parquet_tall_file_12(parquet_test_dir):
    # Define the dataset
    data = {
        "x": range(11, 21),  # Index column
        "y": range(31, 41),  # Index column
        "z": range(41, 51),  # Index column
        "a": [f"val{i}" for i in range(11, 21)],  # Supplementary column
        "b": [i * 2 for i in range(11, 21)],  # Supplementary column
        "c": [i % 3 for i in range(11, 21)],  # Supplementary column
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a Parquet file
    file_path = parquet_test_dir / "test_data_12.parquet"
    df.to_parquet(file_path, engine="pyarrow", row_group_size=5, index=False)

    # Return the path to the Parquet file
    return file_path


@pytest.fixture(scope="session")
def parquet_tall_file_13(parquet_test_dir):
    # Define the dataset with an additional column
    data = {
        "x": range(21, 31),  # Index column
        "y": range(41, 51),  # Index column
        "z": range(51, 61),  # Index column
        "a": [f"val{i}" for i in range(21, 31)],  # Supplementary column
        "b": [i * 2 for i in range(21, 31)],  # Supplementary column
        "c": [i % 3 for i in range(21, 31)],  # Supplementary column
        "d": [f"extra{i}" for i in range(21, 31)],  # Additional column
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a Parquet file
    file_path = parquet_test_dir / "test_data_13.parquet"
    df.to_parquet(file_path, engine="pyarrow", row_group_size=5, index=False)

    # Return the path to the Parquet file
    return file_path


@pytest.fixture(scope="session")
def parquet_unsorted_file(parquet_test_dir):
    # Define the dataset
    data = {
        "x": [3, 1, 2, 5, 4],  # Unsorted column
        "y": [15, 11, 13, 19, 17],  # Unsorted column
        "z": [25, 21, 23, 29, 27],  # Unsorted column
        "h": ["val3", "val1", "val2", "val5", "val4"],  # Supplementary column
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a Parquet file
    file_path = parquet_test_dir / "unsorted_data.parquet"
    df.to_parquet(file_path, engine="pyarrow", row_group_size=5, index=False)

    # Return the path to the Parquet file
    return file_path
