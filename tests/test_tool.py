import pyarrow as pa
import pytest

from parquery import df_to_natural_name, df_to_original_name

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def test_pyarrow_to_natural_name():
    """Test converting PyArrow Table column names to natural format."""
    table = pa.table({"col-1": [1, 2, 3], "col-2": [4, 5, 6], "normal": [7, 8, 9]})

    result = df_to_natural_name(table)

    assert result.column_names == ["col_n_1", "col_n_2", "normal"]
    # Verify it's a new table (immutable)
    assert result is not table
    assert table.column_names == ["col-1", "col-2", "normal"]


def test_pyarrow_to_original_name():
    """Test converting PyArrow Table column names to original format."""
    table = pa.table(
        {"col_n_1": [1, 2, 3], "col_n_2": [4, 5, 6], "normal": [7, 8, 9]}
    )

    result = df_to_original_name(table)

    assert result.column_names == ["col-1", "col-2", "normal"]
    # Verify it's a new table (immutable)
    assert result is not table
    assert table.column_names == ["col_n_1", "col_n_2", "normal"]


def test_pyarrow_roundtrip():
    """Test roundtrip conversion with PyArrow Table."""
    original = pa.table({"col-1": [1, 2], "col-2": [3, 4]})

    natural = df_to_natural_name(original)
    assert natural.column_names == ["col_n_1", "col_n_2"]

    back = df_to_original_name(natural)
    assert back.column_names == ["col-1", "col-2"]
    assert back.equals(original)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_pandas_to_natural_name():
    """Test converting pandas DataFrame column names to natural format."""
    df = pd.DataFrame({"col-1": [1, 2, 3], "col-2": [4, 5, 6], "normal": [7, 8, 9]})
    original_id = id(df)

    result = df_to_natural_name(df)

    # pandas modifies in-place
    assert result is df
    assert id(result) == original_id
    assert list(df.columns) == ["col_n_1", "col_n_2", "normal"]


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_pandas_to_original_name():
    """Test converting pandas DataFrame column names to original format."""
    df = pd.DataFrame(
        {"col_n_1": [1, 2, 3], "col_n_2": [4, 5, 6], "normal": [7, 8, 9]}
    )
    original_id = id(df)

    result = df_to_original_name(df)

    # pandas modifies in-place
    assert result is df
    assert id(result) == original_id
    assert list(df.columns) == ["col-1", "col-2", "normal"]


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_to_natural_name():
    """Test converting Polars DataFrame column names to natural format."""
    df = pl.DataFrame({"col-1": [1, 2, 3], "col-2": [4, 5, 6], "normal": [7, 8, 9]})

    result = df_to_natural_name(df)

    assert result.columns == ["col_n_1", "col_n_2", "normal"]
    # Verify it's a new DataFrame (immutable)
    assert result is not df
    assert df.columns == ["col-1", "col-2", "normal"]


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_to_original_name():
    """Test converting Polars DataFrame column names to original format."""
    df = pl.DataFrame(
        {"col_n_1": [1, 2, 3], "col_n_2": [4, 5, 6], "normal": [7, 8, 9]}
    )

    result = df_to_original_name(df)

    assert result.columns == ["col-1", "col-2", "normal"]
    # Verify it's a new DataFrame (immutable)
    assert result is not df
    assert df.columns == ["col_n_1", "col_n_2", "normal"]


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_roundtrip():
    """Test roundtrip conversion with Polars DataFrame."""
    original = pl.DataFrame({"col-1": [1, 2], "col-2": [3, 4]})

    natural = df_to_natural_name(original)
    assert natural.columns == ["col_n_1", "col_n_2"]

    back = df_to_original_name(natural)
    assert back.columns == ["col-1", "col-2"]
    assert back.equals(original)


def test_invalid_type():
    """Test that invalid types raise TypeError."""
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        df_to_natural_name([1, 2, 3])

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        df_to_original_name({"col": [1, 2, 3]})
