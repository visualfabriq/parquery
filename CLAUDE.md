# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ParQuery is a query and aggregation framework for Parquet files, enabling fast big data aggregations. It provides an OLAP-style approach with Dimensions and Measures, using PyArrow for data operations. Pandas and numpy are optional dependencies for DataFrame support.

## Development Environment

### Python Environment
- **Python Version**: 3.7+ (supports 3.7 and 3.11+)
- **Virtual Environment**: Project uses `.venv` directory
  - Create: `uv venv`
  - Activate: `source .venv/bin/activate`
- **Package Manager**: Use `uv` for all Python package operations (not pip)
- **Type Hints**: Full Python 3.11+ style type hints using `from __future__ import annotations`
  - Uses modern syntax: `list[str]`, `dict[str, str]`, `str | None` instead of typing imports
  - TYPE_CHECKING blocks used for optional pandas imports to avoid circular dependencies

### Installation & Build
```bash
# Build the package in place
python setup.py build_ext --inplace

# Install the package (PyArrow only)
python setup.py install

# Install with optional pandas/numpy support
uv pip install .[optional]

# Install with test dependencies
uv pip install .[test]
```

### Dependencies
- **Core**: pyarrow>=22.0.0 (only required dependency)
- **Optional**: numpy, pandas (version depends on Python version), polars>=0.19.0 - for DataFrame support
- **Testing**: pytest, coverage

## Testing

### Running Tests
```bash
# Run all tests
pytest tests

# Run specific test file
pytest tests/test_parquery.py

# Run with coverage
python -m coverage run -m pytest tests
python -m coverage xml -o cobertura.xml
```

### Code Quality

**Type Checking**: The codebase includes comprehensive type hints. To run type checking:
```bash
# Install mypy if needed
uv pip install mypy

# Run type checker
mypy parquery
```

**Linting & Formatting**: The project uses ruff for linting and formatting:
```bash
# Check for linting issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

All code follows ruff's default standards including:
- No lambda expressions assigned to variables (use def instead)
- No unused variables
- Proper import ordering
- Consistent formatting

### Test Structure
Tests use pytest with class-based organization. Test files:
- `tests/test_parquery.py`: Main aggregation and query tests
- `tests/test_serialization.py`: Serialization/deserialization tests

## Architecture

### Core Modules

**parquery/aggregate.py** - Main aggregation engine
- `aggregate_pq()`: Primary function for querying and aggregating Parquet files
- `finalize_group_by()`: Combines and finalizes grouped results
- Implements push-down filtering using Parquet metadata for performance
- Uses PyArrow compute for filtering and aggregation operations
- Row group-level processing for memory efficiency

**parquery/transport.py** - Serialization utilities
- `serialize_pa_table_bytes()` / `deserialize_pa_table_bytes()`: Binary serialization (bytes)
- `serialize_pa_table_base64()` / `deserialize_pa_table_base64()`: Base64 string serialization (for JSON/SQS)

**parquery/write.py** - Parquet writing utilities
- `df_to_parquet()`: Writes pandas/Polars DataFrames or PyArrow Tables to Parquet with ZSTD compression
  - pandas DataFrames: chunked writing for memory efficiency
  - Polars DataFrames: direct Arrow conversion (very efficient)
  - PyArrow Tables: direct write

**parquery/tool.py** - Utility functions for column name conversion
- `df_to_natural_name()`: Convert column names from original to natural format (replaces '-' with '_n_')
  - pandas: modifies in-place and returns DataFrame
  - Polars: returns new DataFrame (immutable)
  - PyArrow: returns new Table (immutable)
- `df_to_original_name()`: Convert column names from natural to original format (replaces '_n_' with '-')
  - pandas: modifies in-place and returns DataFrame
  - Polars: returns new DataFrame (immutable)
  - PyArrow: returns new Table (immutable)

### Key Design Patterns

**OLAP Aggregation Model**
- Dimensions: Columns to group by (e.g., `groupby_cols`)
- Measures: Columns to aggregate with operations (sum, mean, std, count, etc.)
- Results presented as grouped aggregations over dimensions

**Push-Down Filtering**
- Metadata-based filtering at row group level before reading data
- PyArrow compute expressions for in-memory filtering
- Balances between Parquet statistics and in-memory evaluation

**Pre-aggregation Optimization**
- For "safe" operations (min, max, sum, one), pre-aggregate at row group level
- Reduces memory usage and improves performance for large files
- Final aggregation combines pre-aggregated results

**Memory Management**
- Row group-level processing to handle files larger than memory
- Garbage collection strategy:
  - **Reactive**: gc.collect() on OSError (memory pressure during file/row group reads)
  - **Proactive**: gc.collect() after major operations:
    - After concatenating row group tables (aggregate.py:193)
    - After chunked DataFrame writing (write.py:155)
- Chunked writing for large DataFrames

**Performance Optimizations (20250111)**
- Fixed O(N²) issue in `add_missing_columns_to_table()` by creating all missing columns at once instead of iteratively appending
- Converted `.format()` string formatting to f-strings for better performance
- Added proactive gc.collect() after major memory operations (concat tables, chunked writes) to reduce peak memory usage

### Filter Operations
Supported operators in `data_filter`:
- `in`, `not in`, `nin` (list of values)
- `=`, `==`, `!=`, `>`, `>=`, `<`, `<=` (single value)

Filter structure: `[[column, operator, value(s)], ...]`

**Type Safety**: The library provides type aliases for better IDE support and type checking:
- `FilterOperator`: Literal type for all valid filter operators
- `FilterCondition`: Type alias for (column, operator, value) tuples
- `DataFilter`: Type alias for filter lists

Example with type hints:
```python
from parquery import aggregate_pq, FilterCondition

# Type-safe filter definition
filters: list[FilterCondition] = [
    ("year", ">=", 2020),
    ("status", "in", ["active", "pending"])
]
result = aggregate_pq("data.parquet", ["id"], ["value"], data_filter=filters)
```

### Aggregation Operations
Supported in `aggregation_list`:
- `sum`, `mean`, `std`, `count`, `count_na`, `count_distinct`, `sorted_count_distinct`

Aggregation format options:
1. `['m1', 'm2']` - sum with same column names
2. `[['m1', 'sum'], ['m2', 'count']]` - specify operation
3. `[['m1', 'sum', 'm1_sum']]` - specify output column name

## CI/CD

The project uses CircleCI for CI/CD:
- Tests run on Python 3.11
- Coverage reporting to Codacy
- Builds and publishes to CodeArtifact on master/main branch
- Pre-release versions for UAT branch

## Common Patterns

**Missing Data Handling**
- Missing measure columns: filled with 0.0
- Missing dimension columns: filled with -1 (configurable via `standard_missing_id`)
- Missing files: returns empty DataFrame if `handle_missing_file=True`

**Column Name Management**
- All intermediate operations use input column names
- Output columns renamed at final step based on aggregation list
- Unneeded filter columns dropped after filtering

**PyArrow vs Pandas**
- Internal operations use PyArrow tables for performance
- `as_df` parameter controls return type in `aggregate_pq()`:
  - `None` (default): Smart default - returns DataFrame if pandas installed, else PyArrow Table
  - `True`: Always returns Pandas DataFrame (requires pandas to be installed)
  - `False`: Always returns PyArrow Table (no pandas needed)
- This smart default ensures backward compatibility while allowing pandas-free operation
- Pandas/numpy are optional dependencies; install with `uv pip install 'parquery[optional]'`
- `df_to_parquet()` accepts both pandas DataFrames and PyArrow Tables

## Date Format
Use ISO YYYYMMDD format for dates in comments and documentation.
