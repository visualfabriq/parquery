import pyarrow as pa
import pytest

from parquery import HAS_DUCKDB
from parquery import aggregate_pq
from parquery import aggregate_pq_stream
from parquery import df_to_parquet

ENGINES = ["pyarrow"] + (["duckdb"] if HAS_DUCKDB else [])


def make_path(tmp_path):
    path = tmp_path / "data.parquet"
    df_to_parquet(
        pa.table(
            {
                "group": ["zero", "partial", "cancel", "nonzero"],
                "m1": [0, 2, 1, -3],
                "m2": [0, 0, -1, 0],
            }
        ),
        str(path),
    )
    return path


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_filter_removes_only_all_zero_groups(tmp_path, engine):
    path = make_path(tmp_path)
    result = aggregate_pq(str(path), ["group"], ["m1", "m2"], engine=engine, as_df=False).sort_by("group")
    assert result.to_pydict() == {
        "group": ["cancel", "nonzero", "partial"],
        "m1": [1, -3, 2],
        "m2": [-1, 0, 0],
    }


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_filter_can_be_disabled(tmp_path, engine):
    result = aggregate_pq(
        str(make_path(tmp_path)),
        ["group"],
        ["m1"],
        engine=engine,
        as_df=False,
        zero_filter=False,
    )
    assert result.num_rows == 4


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_filter_supports_aliases_and_non_sum_operations(tmp_path, engine):
    path = make_path(tmp_path)
    result = aggregate_pq(
        str(path),
        ["group"],
        [["m1", "mean", "average"], ["m2", "max", "peak"]],
        engine=engine,
        as_df=False,
    )
    expected_groups = {"partial", "cancel", "nonzero"}
    if engine == "duckdb":
        expected_groups.add("zero")
    assert set(result["group"].to_pylist()) == expected_groups
    assert result.column_names == ["group", "average", "peak"]


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB is required")
def test_zero_filter_streaming(tmp_path):
    result = pa.Table.from_batches(list(aggregate_pq_stream(str(make_path(tmp_path)), ["group"], ["m1", "m2"])))
    assert set(result["group"].to_pylist()) == {"partial", "cancel", "nonzero"}


def test_zero_filter_does_not_apply_to_raw_rows(tmp_path):
    path = make_path(tmp_path)
    result = aggregate_pq(str(path), ["group"], ["m1"], aggregate=False, as_df=False)
    assert result.num_rows == 4


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_filter_handles_cancellation_and_prefilled_missing_values(tmp_path, engine):
    path = tmp_path / "cancel.parquet"
    # Business data is normalized before it reaches ParQuery: missing numeric
    # measures are represented as 0.0 rather than NULL.
    df_to_parquet(
        pa.table(
            {
                "group": ["cancel", "cancel", "missing", "live"],
                "measure": [1.0, -1.0, 0.0, 2.0],
            }
        ),
        str(path),
    )
    result = aggregate_pq(str(path), ["group"], ["measure"], engine=engine, as_df=False)
    result = result.sort_by("group")
    expected_groups = ["live"] if engine == "pyarrow" else ["cancel", "live"]
    assert result["group"].to_pylist() == expected_groups
    assert result["measure"].to_pylist() == ([2.0] if engine == "pyarrow" else [0.0, 2.0])


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_filter_handles_500_measures(tmp_path, engine):
    path = tmp_path / "many.parquet"
    values = {f"m{i}": [0] for i in range(500)}
    values["m499"] = [1]
    values["group"] = ["live"]
    df_to_parquet(pa.table(values), str(path))
    measures = [f"m{i}" for i in range(500)]
    result = aggregate_pq(str(path), ["group"], measures, engine=engine, as_df=False)
    assert result.num_rows == 1
    assert result["m499"].to_pylist() == [1]
