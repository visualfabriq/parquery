import tempfile

import pyarrow.parquet as pq
import pytest

from parquery import aggregate_pq, df_to_parquet

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_write_and_read():
    """Test writing Polars DataFrame to Parquet and reading it back."""
    # Create a simple Polars DataFrame
    df = pl.DataFrame(
        {
            "f0": ["a", "a", "b", "b", "c", "c"],
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [10, 20, 30, 40, 50, 60],
        }
    )

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        filename = tmp.name

    try:
        # Write using df_to_parquet
        df_to_parquet(df, filename)

        # Read back using PyArrow to verify
        table = pq.read_table(filename)
        assert table.num_rows == 6
        assert table.column_names == ["f0", "f1", "f2"]

        # Test aggregation on the written file
        result = aggregate_pq(filename, ["f0"], ["f2"], as_df=False)
        assert result.num_rows == 3
        assert result.column_names == ["f0", "f2"]

        # Verify aggregated values
        result_dict = {row["f0"]: row["f2"] for row in result.to_pylist()}
        assert result_dict == {"a": 30, "b": 70, "c": 110}

    finally:
        import os

        if os.path.exists(filename):
            os.remove(filename)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_large_dataframe():
    """Test writing a larger Polars DataFrame."""
    # Create a larger DataFrame
    df = pl.DataFrame(
        {
            "group": ["A"] * 5000 + ["B"] * 5000 + ["C"] * 5000,
            "value": list(range(15000)),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        filename = tmp.name

    try:
        # Write
        df_to_parquet(df, filename)

        # Aggregate
        result = aggregate_pq(
            filename, ["group"], [["value", "sum"]], as_df=False
        )
        assert result.num_rows == 3

        # Verify sums
        result_dict = {row["group"]: row["value"] for row in result.to_pylist()}
        expected_a = sum(range(5000))
        expected_b = sum(range(5000, 10000))
        expected_c = sum(range(10000, 15000))
        assert result_dict == {"A": expected_a, "B": expected_b, "C": expected_c}

    finally:
        import os

        if os.path.exists(filename):
            os.remove(filename)
