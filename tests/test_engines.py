"""
Tests for engine selection and comparison between DuckDB and PyArrow engines.
"""

import os
import tempfile

import pyarrow as pa
import pytest

from parquery import HAS_DUCKDB, aggregate_pq, df_to_parquet


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
        import math

        # Ensure we're comparing PyArrow Tables
        assert isinstance(pyarrow_result, pa.Table), "PyArrow result should be a Table"
        assert isinstance(duckdb_result, pa.Table), "DuckDB result should be a Table"

        # Sort both tables by first column for consistent ordering
        if pyarrow_result.num_rows > 0:
            first_col = pyarrow_result.column_names[0]
            indices_pa = pa.compute.sort_indices(pyarrow_result[first_col])
            pyarrow_result = pa.compute.take(pyarrow_result, indices_pa)

            indices_db = pa.compute.sort_indices(duckdb_result[first_col])
            duckdb_result = pa.compute.take(duckdb_result, indices_db)

        # Check shape
        assert pyarrow_result.num_rows == duckdb_result.num_rows, (
            f"Row count mismatch: {pyarrow_result.num_rows} vs {duckdb_result.num_rows}"
        )
        assert pyarrow_result.num_columns == duckdb_result.num_columns, (
            f"Column count mismatch: {pyarrow_result.num_columns} vs {duckdb_result.num_columns}"
        )

        # Check column names
        assert pyarrow_result.column_names == duckdb_result.column_names, (
            f"Column names mismatch: {pyarrow_result.column_names} vs {duckdb_result.column_names}"
        )

        # Compare values column by column
        for col_name in pyarrow_result.column_names:
            pa_col = pyarrow_result[col_name].to_pylist()
            db_col = duckdb_result[col_name].to_pylist()

            for i, (pa_val, db_val) in enumerate(zip(pa_col, db_col)):
                # Handle None/null values
                if pa_val is None and db_val is None:
                    continue
                if pa_val is None or db_val is None:
                    raise AssertionError(
                        f"Null mismatch in column '{col_name}' row {i}"
                    )

                # Compare floats with tolerance
                if isinstance(pa_val, (float, int)) and isinstance(
                    db_val, (float, int)
                ):
                    if not math.isclose(
                        float(pa_val),
                        float(db_val),
                        rel_tol=10 ** (-decimal),
                        abs_tol=10 ** (-decimal),
                    ):
                        raise AssertionError(
                            f"Value mismatch in column '{col_name}' row {i}: "
                            f"{pa_val} != {db_val}"
                        )
                else:
                    # Direct comparison for non-numeric types
                    assert pa_val == db_val, (
                        f"Value mismatch in column '{col_name}' row {i}: {pa_val} != {db_val}"
                    )

    def test_simple_sum(self, test_parquet_file):
        """Test simple sum aggregation."""
        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            engine="pyarrow",
            as_df=False,
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            engine="duckdb",
            as_df=False,
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

    def test_multiple_aggregations(self, test_parquet_file):
        """Test multiple aggregations on different columns."""
        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"], ["m2", "mean"], ["f0", "count"]],
            engine="pyarrow",
            as_df=False,
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"], ["m2", "mean"], ["f0", "count"]],
            engine="duckdb",
            as_df=False,
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
            as_df=False,
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            data_filter=data_filter,
            engine="duckdb",
            as_df=False,
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
            as_df=False,
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum"]],
            data_filter=data_filter,
            engine="duckdb",
            as_df=False,
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

    def test_min_max(self, test_parquet_file):
        """Test min/max aggregations."""
        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "min"], ["m2", "max"]],
            engine="pyarrow",
            as_df=False,
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "min"], ["m2", "max"]],
            engine="duckdb",
            as_df=False,
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

    def test_custom_output_names(self, test_parquet_file):
        """Test aggregations with custom output names."""
        pyarrow_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum", "total_m1"], ["m2", "mean", "avg_m2"]],
            engine="pyarrow",
            as_df=False,
        )
        duckdb_result = aggregate_pq(
            test_parquet_file,
            ["group_id"],
            [["m1", "sum", "total_m1"], ["m2", "mean", "avg_m2"]],
            engine="duckdb",
            as_df=False,
        )
        self.assert_results_equal(pyarrow_result, duckdb_result)

        # Check column names (use column_names for PyArrow Tables)
        assert "total_m1" in pyarrow_result.column_names
        assert "avg_m2" in pyarrow_result.column_names
        assert "total_m1" in duckdb_result.column_names
        assert "avg_m2" in duckdb_result.column_names


class TestEngineBackwardCompatibility:
    """Test that existing code continues to work."""

    def test_default_behavior_unchanged(self, test_parquet_file):
        """Test that default behavior (no engine param) works."""
        result = aggregate_pq(
            test_parquet_file, ["group_id"], [["m1", "sum"]], as_df=False
        )
        assert result.num_rows == 3

    def test_pyarrow_direct_call(self, test_parquet_file):
        """Test call to aggregate_pq with explicit pyarrow engine (returns PyArrow Table)."""
        result = aggregate_pq(
            test_parquet_file, ["group_id"], ["m1"], engine="pyarrow", as_df=False
        )
        assert isinstance(result, pa.Table)
        assert result.num_rows == 3

    @pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")
    def test_duckdb_direct_call(self, test_parquet_file):
        """Test call to aggregate_pq with explicit duckdb engine (returns PyArrow Table)."""
        result = aggregate_pq(
            test_parquet_file, ["group_id"], ["m1"], engine="duckdb", as_df=False
        )
        assert isinstance(result, pa.Table)
        assert result.num_rows == 3
