"""Regression tests for DuckDB read hardening.

Two independent failure modes on the shared EFS data lake:

* A concurrent writer ``os.replace``-ing a shard mid-read could splice two file
  versions into a corrupt-but-valid result. ``_aggregate_pinned`` reads through
  ``/dev/fd/<fd>`` so the whole read sees one consistent inode snapshot.
* By default DuckDB sizes memory/threads to the host, not the container.
  ``call_duckdb`` maps the ``DUCKDB_*`` env vars onto the connection config so
  the deploying service can tighten memory/threads and enable disk spill.
"""

import logging
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


def _result_map(res):
    """Map groupby key -> summed measure, so assertions check values, not just keys."""
    return dict(zip(res.column("a-31").to_pylist(), res.column("m1").to_pylist(), strict=True))


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

    # Both keys AND the summed measure reflect the pre-rename snapshot.
    assert _result_map(res) == {20251201: 1.0, 20251202: 1.0}


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

    assert _result_map(res) == {99999999: 1.0}


def test_fallback_without_dev_fd_returns_correct_result(tmp_path, monkeypatch, caplog):
    # The /dev/fd-absent branch must still return a correct result on the normal
    # (no concurrent rename) path, and must warn that protection is degraded.
    target = str(tmp_path / "shard.parquet")
    _write(target, [20251201, 20251202])

    monkeypatch.setattr(adb, "_CAN_PIN_FD", False)
    monkeypatch.setattr(adb, "_fd_fallback_warned", False)

    with caplog.at_level(logging.WARNING):
        res = aggregate_pq(target, ["a-31"], [["m1", "sum"]], engine="duckdb", as_df=False, aggregate=True)

    assert _result_map(res) == {20251201: 1.0, 20251202: 1.0}
    assert any("without inode pinning" in r.message for r in caplog.records)


@pytest.mark.skipif(not adb._CAN_PIN_FD, reason="/dev/fd not available")
def test_oserror_retried_once_with_fresh_fd(tmp_path, monkeypatch):
    target = str(tmp_path / "shard.parquet")
    _write(target, [20251201, 20251202])

    real_call = adb.call_duckdb
    calls = {"n": 0}

    def flaky(sql):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient stale handle")
        return real_call(sql)

    opened, closed = [], []
    real_open, real_close = os.open, os.close

    def counting_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened.append(fd)
        return fd

    def counting_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(adb, "call_duckdb", flaky)
    monkeypatch.setattr(os, "open", counting_open)
    monkeypatch.setattr(os, "close", counting_close)

    res = aggregate_pq(target, ["a-31"], [["m1", "sum"]], engine="duckdb", as_df=False, aggregate=True)

    assert calls["n"] == 2  # failed once, retried once
    assert len(opened) == 2  # a fresh fd opened per attempt
    assert opened == closed  # every fd closed, including the failing attempt (no leak)
    assert _result_map(res) == {20251201: 1.0, 20251202: 1.0}


def test_oserror_second_failure_propagates(tmp_path, monkeypatch):
    target = str(tmp_path / "shard.parquet")
    _write(target, [20251201])

    def always_raise(sql):
        raise OSError("persistent")

    monkeypatch.setattr(adb, "call_duckdb", always_raise)

    with pytest.raises(OSError, match="persistent"):
        aggregate_pq(target, ["a-31"], [["m1", "sum"]], engine="duckdb", as_df=False, aggregate=True)


def test_connection_config_from_env(tmp_path, monkeypatch):
    import duckdb

    captured = {}
    real_connect = duckdb.connect

    def fake_connect(database, config=None):
        captured["config"] = dict(config or {})
        return real_connect(database, config=config or {})

    spill = str(tmp_path / "spill")
    monkeypatch.setattr(adb, "DUCKDB_MEMORY_LIMIT", "512MB")
    monkeypatch.setattr(adb, "DUCKDB_THREADS", 2)  # int, coerced at import
    monkeypatch.setattr(adb, "DUCKDB_TEMP_DIR", spill)
    monkeypatch.setattr(adb, "DUCKDB_MAX_TEMP_DIR_SIZE", "4GB")
    monkeypatch.setattr(duckdb, "connect", fake_connect)

    adb.call_duckdb("SELECT 1")

    cfg = captured["config"]
    assert cfg["memory_limit"] == "512MB"
    assert cfg["threads"] == 2
    # temp_directory is a per-connection subdir under the configured root, not the
    # root itself, so concurrent connections cannot collide on one spill file
    # (FIN-4849).
    assert cfg["temp_directory"] != spill
    assert os.path.dirname(cfg["temp_directory"]) == spill
    assert cfg["max_temp_directory_size"] == "4GB"
    assert os.path.isdir(spill)  # root created if missing


def test_spill_dir_unique_per_connection_and_cleaned_up(tmp_path, monkeypatch):
    # Without isolation every :memory: connection spills to the same
    # ``duckdb_temp_storage_DEFAULT-0.tmp`` under a shared temp_directory; under
    # concurrency those paths collide and clobber each other, yielding IO errors
    # ("Could not read enough bytes") or silently wrong aggregations. Each
    # connection must get its own spill subdir, removed once the query finishes.
    # Regression for FIN-4849.
    import duckdb

    spill = str(tmp_path / "spill")
    seen = []
    real_connect = duckdb.connect

    def capturing_connect(database, config=None):
        temp_dir = (config or {}).get("temp_directory")
        assert temp_dir is not None
        assert os.path.isdir(temp_dir)  # created before connect
        assert os.path.dirname(temp_dir) == spill
        seen.append(temp_dir)
        return real_connect(database, config=config or {})

    monkeypatch.setattr(adb, "DUCKDB_TEMP_DIR", spill)
    monkeypatch.setattr(duckdb, "connect", capturing_connect)

    adb.call_duckdb("SELECT 1")
    adb.call_duckdb("SELECT 1")

    assert len(seen) == 2
    assert seen[0] != seen[1]  # distinct spill paths -> no collision
    assert all(d.startswith(os.path.join(spill, f"{os.getpid()}-")) for d in seen)
    assert not any(os.path.exists(d) for d in seen)  # cleaned up after each query
    assert os.listdir(spill) == []  # root retained, no leaked subdirs


def test_connection_config_threads_only(monkeypatch):
    # Partial config: max-temp-size without a temp dir is dropped, and no spill
    # directory is created when only threads is set.
    import duckdb

    captured = {}
    real_connect = duckdb.connect

    def fake_connect(database, config=None):
        captured["config"] = dict(config or {})
        return real_connect(database, config=config or {})

    monkeypatch.setattr(adb, "DUCKDB_MEMORY_LIMIT", None)
    monkeypatch.setattr(adb, "DUCKDB_THREADS", 2)
    monkeypatch.setattr(adb, "DUCKDB_TEMP_DIR", None)
    monkeypatch.setattr(adb, "DUCKDB_MAX_TEMP_DIR_SIZE", "4GB")
    monkeypatch.setattr(duckdb, "connect", fake_connect)

    adb.call_duckdb("SELECT 1")

    assert captured["config"] == {"threads": 2}


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


def test_int_env_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("DUCKDB_THREADS", "two")
    with pytest.raises(ValueError, match="DUCKDB_THREADS must be an integer"):
        adb._int_env("DUCKDB_THREADS")


def test_int_env_none_when_unset(monkeypatch):
    monkeypatch.delenv("DUCKDB_THREADS", raising=False)
    assert adb._int_env("DUCKDB_THREADS") is None


def test_call_duckdb_closes_connection_on_failure(monkeypatch):
    # The connection's memory must be released even when the query raises, so a
    # subsequent retry (and its gc.collect) is not competing with a live conn.
    import duckdb

    closed = {"n": 0}

    class FailingConn:
        def execute(self, sql):
            raise RuntimeError("boom")

        def close(self):
            closed["n"] += 1

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: FailingConn())

    with pytest.raises(RuntimeError, match="boom"):
        adb.call_duckdb("SELECT 1")

    assert closed["n"] == 1
