import pyarrow as pa
import pytest

from parquery import HAS_DUCKDB
from parquery import aggregate_pq
from parquery import aggregate_pq_stream
from parquery import df_to_parquet


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_aggregate_pq_stream_matches_table(tmp_path):
    path = tmp_path / "data.parquet"
    df_to_parquet(pa.table({"group": ["a", "a", "b"], "value": [1, 2, 4]}), str(path))

    batches = list(aggregate_pq_stream(str(path), ["group"], [["value", "sum", "total"]], batch_size=1))
    streamed = pa.Table.from_batches(batches)
    expected = aggregate_pq(str(path), ["group"], [["value", "sum", "total"]], as_df=False, engine="duckdb")

    assert isinstance(
        reader := aggregate_pq_stream(str(path), ["group"], [["value", "sum", "total"]]), pa.RecordBatchReader
    )
    assert reader.schema.names == expected.column_names
    assert streamed.column_names == expected.column_names
    assert streamed.sort_by("group") == expected.sort_by("group")


def write_test_table(tmp_path, table):
    path = tmp_path / "data.parquet"
    df_to_parquet(table, str(path))
    return path


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_filter_and_custom_aggregations(tmp_path):
    path = write_test_table(
        tmp_path,
        pa.table(
            {
                "group": ["a", "a", "b", "b"],
                "value": [1, 2, 10, 20],
                "status": ["ok", "skip", "ok", "ok"],
            }
        ),
    )

    reader = aggregate_pq_stream(
        str(path),
        ["group"],
        [["value", "sum", "total"], ["value", "count", "rows"]],
        data_filter=[["status", "==", "ok"]],
        batch_size=1,
    )
    result = pa.Table.from_batches(list(reader))

    assert result.column_names == ["group", "total", "rows"]
    assert result.sort_by("group").to_pydict() == {
        "group": ["a", "b"],
        "total": [1, 30],
        "rows": [1, 2],
    }


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_without_groupby_matches_normal_api(tmp_path):
    path = write_test_table(tmp_path, pa.table({"value": [1, 2, 3]}))

    streamed = pa.Table.from_batches(list(aggregate_pq_stream(str(path), [], [["value", "sum", "total"]])))
    normal = aggregate_pq(str(path), [], [["value", "sum", "total"]], as_df=False, engine="duckdb")
    assert streamed == normal


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_raw_rows_matches_normal_api(tmp_path):
    path = write_test_table(tmp_path, pa.table({"zeta": ["a", "b"], "alpha": [1, 2]}))

    streamed = pa.Table.from_batches(list(aggregate_pq_stream(str(path), ["zeta"], ["alpha"], aggregate=False)))
    normal = aggregate_pq(str(path), ["zeta"], ["alpha"], aggregate=False, as_df=False, engine="duckdb")
    assert streamed.column_names == normal.column_names
    assert streamed.sort_by("zeta") == normal.sort_by("zeta")


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_missing_requested_columns_use_normal_defaults(tmp_path):
    path = write_test_table(tmp_path, pa.table({"value": [1, 2]}))

    reader = aggregate_pq_stream(str(path), ["missing_group"], [["missing_value", "sum"]])
    assert list(reader) == []


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_invalid_engine_is_rejected(tmp_path):
    path = write_test_table(tmp_path, pa.table({"value": [1]}))
    with pytest.raises(ValueError, match="requires the DuckDB engine"):
        aggregate_pq_stream(str(path), [], ["value"], engine="banana")


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_batches_respect_batch_size(tmp_path):
    path = write_test_table(tmp_path, pa.table({"value": list(range(7))}))
    batches = list(aggregate_pq_stream(str(path), [], ["value"], aggregate=False, batch_size=2))
    assert len(batches) > 1
    assert sum(batch.num_rows for batch in batches) == 7


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_reader_close_cleans_up_early(tmp_path):
    path = write_test_table(tmp_path, pa.table({"value": list(range(10))}))
    reader = aggregate_pq_stream(str(path), [], ["value"], aggregate=False, batch_size=2)
    next(reader)
    reader.close()
    assert reader.schema.names == ["value"]


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_missing_filter_column_is_empty(tmp_path):
    path = write_test_table(tmp_path, pa.table({"value": [1, 2]}))

    reader = aggregate_pq_stream(str(path), [], ["value"], data_filter=[["missing", "=", 1]])
    assert isinstance(reader, pa.RecordBatchReader)
    assert reader.schema.names == ["value"]
    assert list(reader) == []


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_streaming_missing_file_behavior(tmp_path):
    missing = str(tmp_path / "missing.parquet")
    reader = aggregate_pq_stream(missing, ["group"], ["value"])
    assert isinstance(reader, pa.RecordBatchReader)
    assert reader.schema.names == ["group", "value"]
    assert list(reader) == []
    with pytest.raises(OSError, match="File not found"):
        list(aggregate_pq_stream(missing, ["group"], ["value"], handle_missing_file=False))
