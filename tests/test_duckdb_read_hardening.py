"""Regression tests for DuckDB read hardening.

Two independent failure modes on the shared EFS data lake:

* A concurrent writer ``os.replace``-ing a shard mid-read could splice two file
  versions into a corrupt-but-valid result. ``_aggregate_pinned`` reads through
  ``/dev/fd/<fd>`` so the whole read sees one consistent inode snapshot.
* DuckDB runs unbounded unless told otherwise. ``call_duckdb`` maps the
  ``DUCKDB_*`` env vars onto the connection config so memory/threads/spill are
  bounded by the deploying service.
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import parquery.aggregate_duckdb as adb
from parquery import HAS_DUCKDB
from parquery import aggregate_pq

pytestmark = pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB not installed")


def _write(path, dates):
    pq.write_table(
        pa.table({"a-31": dates, "g": [1] * len(dates), "m1": [1.0] * len(dates)}),
        path,
    )


def _swap_then_read(real, target, newfile):
    """Wrap call_duckdb to simulate the writer's rename landing mid-read."""

    def wrapper(sql):
        os.replace(newfile, target)
        return real(sql)

    return wrapper


@pytest.mark.skipif(not adb._CAN_PIN_FD, reason="/dev/fd not available")
def test_pinned_read_survives_concurrent_replace(tmp_path, monkeypatch):
    target = str(tmp_path / "shard.parquet")
    _write(target, [20251201, 20251202])  # consistent OLD content
    newfile = str(tmp_path / "new.parquet")
    _write(newfile, [99999999])  # different content the rename would expose

    # The fd is opened in _aggregate_pinned before call_duckdb runs, so the
    # rename inside the wrapper cannot change what /dev/fd/<fd> resolves to.
    monkeypatch.setattr(adb, "call_duckdb", _swap_then_read(adb.call_duckdb, target, newfile))

    res = aggregate_pq(target, ["a-31"], [["m1", "sum"]], engine="duckdb", as_df=False, aggregate=True)

    assert sorted(res.column("a-31").to_pylist()) == [20251201, 20251202]


def test_without_pin_concurrent_replace_is_visible(tmp_path, monkeypatch):
    # Control: with pinning disabled, the same rename leaks the new file. Proves
    # the pin (not test setup) is what isolates the read above.
    target = str(tmp_path / "shard.parquet")
    _write(target, [20251201, 20251202])
    newfile = str(tmp_path / "new.parquet")
    _write(newfile, [99999999])

    monkeypatch.setattr(adb, "_CAN_PIN_FD", False)
    monkeypatch.setattr(adb, "call_duckdb", _swap_then_read(adb.call_duckdb, target, newfile))

    res = aggregate_pq(target, ["a-31"], [["m1", "sum"]], engine="duckdb", as_df=False, aggregate=True)

    assert sorted(res.column("a-31").to_pylist()) == [99999999]


def test_connection_config_from_env(tmp_path, monkeypatch):
    import duckdb

    captured = {}
    real_connect = duckdb.connect

    def fake_connect(database, config=None):
        captured["config"] = dict(config or {})
        return real_connect(database, config=config or {})

    spill = str(tmp_path / "spill")
    monkeypatch.setattr(adb, "DUCKDB_MEMORY_LIMIT", "512MB")
    monkeypatch.setattr(adb, "DUCKDB_THREADS", "2")
    monkeypatch.setattr(adb, "DUCKDB_TEMP_DIR", spill)
    monkeypatch.setattr(adb, "DUCKDB_MAX_TEMP_DIR_SIZE", "4GB")
    monkeypatch.setattr(duckdb, "connect", fake_connect)

    adb.call_duckdb("SELECT 1")

    cfg = captured["config"]
    assert cfg["memory_limit"] == "512MB"
    assert cfg["threads"] == 2  # coerced to int
    assert cfg["temp_directory"] == spill
    assert cfg["max_temp_directory_size"] == "4GB"
    assert os.path.isdir(spill)  # created if missing


def test_connection_config_empty_when_unset(monkeypatch):
    import duckdb

    captured = {}
    real_connect = duckdb.connect

    def fake_connect(database, config=None):
        captured["config"] = dict(config or {})
        return real_connect(database, config=config or {})

    for name in ("DUCKDB_MEMORY_LIMIT", "DUCKDB_THREADS", "DUCKDB_TEMP_DIR", "DUCKDB_MAX_TEMP_DIR_SIZE"):
        monkeypatch.setattr(adb, name, None)
    monkeypatch.setattr(duckdb, "connect", fake_connect)

    adb.call_duckdb("SELECT 1")

    assert captured["config"] == {}
