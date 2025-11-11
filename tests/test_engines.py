"""
Tests for engine selection and comparison between DuckDB and PyArrow engines.
"""

import os
import tempfile

import pyarrow as pa
import pytest

from parquery import HAS_DUCKDB, aggregate_pq, aggregate_pq_pyarrow, df_to_parquet

try:
    from parquery import aggregate_pq_duckdb

    HAS_DUCKDB_MODULE = True
except ImportError:
    HAS_DUCKDB_MODULE = False


@pytest.fixture
def test_parquet_file():
    """Create a temporary test Parquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.parquet")

        # Create test data
        data = {
            "group_id": [1, 1, 2, 2, 3, 3] * 10,
            "f0": list(range(60)),
            "m1": [float(i) * 1.5 for i in range(60)],
            "m2": [float(i) * 2.0 for i in range(60)],
        }

        table = pa.table(data)
        df_to_parquet(table, file_path)

        yield file_path


class TestEngineSelection:
    """Test engine selection and auto-detection."""

    def test_default_engine_selection(self, test_parquet_file):
        """Test that engine='auto' selects correctly."""
        result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            engine="auto",
            as_df=False,
        )
        assert result.num_rows == 3

    def test_pyarrow_engine_explicit(self, test_parquet_file):
        """Test explicit PyArrow engine selection."""
        result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            engine="pyarrow",
            as_df=False,
        )
        assert result.num_rows == 3

    @pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
    def test_duckdb_engine_explicit(self, test_parquet_file):
        """Test explicit DuckDB engine selection."""
        result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            engine="duckdb",
            as_df=False,
        )
        assert result.num_rows == 3

    def test_invalid_engine(self, test_parquet_file):
        """Test that invalid engine raises ValueError."""
        with pytest.raises(ValueError, match="Unknown engine"):
            aggregate_pq(
                test_parquet_file,
                ["group_id"],
                [["m1", "sum"]],
                engine="invalid",
                as_df=False,
            )

    def test_duckdb_not_installed(self, test_parquet_file):
        """Test that requesting DuckDB when not installed raises ImportError."""
        if HAS_DUCKDB:
            pytest.skip("DuckDB is installed, cannot test ImportError case")

        with pytest.raises(ImportError, match="DuckDB engine requested"):
            aggregate_pq(
                test_parquet_file,
                ["group_id"],
                [["m1", "sum"]],
                engine="duckdb",
                as_df=False,
            )


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
class TestEngineEquivalence:
    """Test that PyArrow and DuckDB engines return equivalent results."""

    def assert_results_equal(self, pyarrow_result, duckdb_result, decimal=5):
        """Assert two results are equal, handling float comparison."""
        # Convert to pandas for easier comparison
        if isinstance(pyarrow_result, pa.Table):
            pyarrow_result = pyarrow_result.to_pandas()
        if isinstance(duckdb_result, pa.Table):
            duckdb_result = duckdb_result.to_pandas()

        # Sort by first column to ensure consistent ordering
        first_col = pyarrow_result.columns[0]
        pyarrow_result = pyarrow_result.sort_values(first_col).reset_index(drop=True)
        duckdb_result = duckdb_result.sort_values(first_col).reset_index(drop=True)

        # Check shape
        assert (
            pyarrow_result.shape == duckdb_result.shape
        ), f"Shape mismatch: {pyarrow_result.shape} vs {duckdb_result.shape}"

        # Compare values with tolerance for floats
        import pandas as pd

        pd.testing.assert_frame_equal(
            pyarrow_result,
            duckdb_result,
            check_dtype=False,  # Allow minor type differences
            rtol=10 ** (-decimal),
            atol=10 ** (-decimal),
        )

    def test_simple_sum(self, test_parquet_file):
        """Test simple sum aggregation."""
        pyarrow_result = aggregate_pq(
            test_parquet_file, ["group_id"], [["m1", "sum"]], engine="pyarrow"
        )
        duckdb_result = aggregate_pq(
            test_parquet_file, ["group_id"], [["m1", "sum"]], engine="duckdb"
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

    def test_multiple_aggregations(self, test_parquet_file):
        """Test multiple aggregations on different columns."""
        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"], ["m2", "mean"], ["f0", "count"]],
            engine="pyarrow",
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"], ["m2", "mean"], ["f0", "count"]],
            engine="duckdb",
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

    def test_with_filter(self, test_parquet_file):
        """Test aggregation with filter."""
        data_filter = [["f0", ">", 30]]

        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            data_filter=data_filter,
            engine="pyarrow",
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            data_filter=data_filter,
            engine="duckdb",
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

    def test_with_in_filter(self, test_parquet_file):
        """Test aggregation with 'in' filter."""
        data_filter = [["group_id", "in", [1, 3]]]

        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            data_filter=data_filter,
            engine="pyarrow",
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            data_filter=data_filter,
            engine="duckdb",
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

    def test_min_max(self, test_parquet_file):
        """Test min/max aggregations."""
        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "min"], ["m2", "max"]],
            engine="pyarrow",
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "min"], ["m2", "max"]],
            engine="duckdb",
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

    def test_custom_output_names(self, test_parquet_file):
        """Test aggregations with custom output names."""
        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum", "total_m1"], ["m2", "mean", "avg_m2"]],
            engine="pyarrow",
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum", "total_m1"], ["m2", "mean", "avg_m2"]],
            engine="duckdb",
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

        # Check column names
        assert "total_m1" in pyarrow_result.columns
        assert "avg_m2" in pyarrow_result.columns
        assert "total_m1" in duckdb_result.columns
        assert "avg_m2" in duckdb_result.columns


class TestEngineBackwardCompatibility:
    """Test that existing code continues to work."""

    def test_default_behavior_unchanged(self, test_parquet_file):
        """Test that default behavior (no engine param) works."""
        result = aggregate_pq(
            test_parquet_file, ["group_id"], [["m1", "sum"]], as_df=False
        )
        assert result.num_rows == 3

    def test_pyarrow_direct_call(self, test_parquet_file):
        """Test direct call to aggregate_pq_pyarrow (always returns PyArrow Table)."""
        result = aggregate_pq_pyarrow(
            test_parquet_file, ["group_id"], [["m1", "sum"]]
        )
        assert isinstance(result, pa.Table)
        assert result.num_rows == 3

    @pytest.mark.skipif(not HAS_DUCKDB_MODULE, reason="DuckDB not installed")
    def test_duckdb_direct_call(self, test_parquet_file):
        """Test direct call to aggregate_pq_duckdb (always returns PyArrow Table)."""
        result = aggregate_pq_duckdb(
            test_parquet_file, ["group_id"], [["m1", "sum"]]
        )
        assert isinstance(result, pa.Table)
        assert result.num_rows == 3
