import itertools as itt
import math
import os
import random
import shutil
import statistics
import tempfile
from contextlib import contextmanager
from importlib.util import find_spec
import pyarrow as pa
import pytest
from parquery import aggregate_pq, df_to_parquet, HAS_DUCKDB

HAS_PANDAS = find_spec("pandas") is not None
ENGINES_TO_TEST = ["pyarrow"]
if HAS_DUCKDB:
    ENGINES_TO_TEST.append("duckdb")


def create_table_from_generator(generator, schema):
    """Convert generator data to PyArrow table."""
    data = list(generator)
    if not data:
        return pa.table({name: [] for name, _ in schema})
    columns = list(zip(*data))
    table_data = {name: list(col) for (name, _), col in zip(schema, columns)}
    return pa.table(table_data, schema=pa.schema(schema))


def assert_tables_equal(table1, table2, decimal=6):
    """Assert two PyArrow tables are equal, with float tolerance."""
    assert table1.column_names == table2.column_names, "Column names don't match"
    assert table1.num_rows == table2.num_rows, "Number of rows don't match"
    for col_name in table1.column_names:
        col1 = table1[col_name].to_pylist()
        col2 = table2[col_name].to_pylist()
        for i, (v1, v2) in enumerate(zip(col1, col2)):
            if isinstance(v1, float) and isinstance(v2, float):
                if not math.isclose(v1, v2, rel_tol=10**-decimal, abs_tol=10**-decimal):
                    raise AssertionError(
                        f"Column '{col_name}' row {i}: {v1} != {v2} (tolerance={10**-decimal})"
                    )
            else:
                assert v1 == v2, f"Column '{col_name}' row {i}: {v1} != {v2}"


def round_float_columns(table, decimal=6):
    """Round float columns in a PyArrow table."""
    arrays = []
    for col_name in table.column_names:
        col = table[col_name]
        if pa.types.is_floating(col.type):
            rounded = [
                (round(v, decimal) if v is not None else None) for v in col.to_pylist()
            ]
            arrays.append(pa.array(rounded, type=col.type))
        else:
            arrays.append(col)
    return pa.table(arrays, names=table.column_names)


def sort_table(table, sort_keys):
    """Sort a PyArrow table by specified columns."""
    if not sort_keys or table.num_rows == 0:
        return table
    # Convert to list of dicts, sort, convert back
    rows = table.to_pylist()
    sorted_rows = sorted(rows, key=lambda x: [x[k] for k in sort_keys])
    return pa.Table.from_pylist(sorted_rows, schema=table.schema)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
class TestParquery(object):
    @contextmanager
    def on_disk_data_cleaner(self, table):
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        yield self.filename
        shutil.rmtree(self.rootdir)
        self.rootdir = None

    def setup_method(self):
        self.filename = None

    def teardown_method(self):
        if self.filename:
            os.remove(self.filename)
            self.filename = None

    @staticmethod
    def gen_dataset_count(N):
        pool = itt.cycle(["a", "a", "b", "b", "b", "c", "c", "c", "c", "c"])
        pool_b = itt.cycle([0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                random.random(),
                random.randint(-10, 10),
                random.randint(-10, 10),
            )
            yield d

    @staticmethod
    def gen_dataset_count_with_NA(N):
        pool = itt.cycle(["a", "a", "b", "b", "b", "c", "c", "c", "c", "c"])
        pool_b = itt.cycle([0.0, 0.1, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_e = itt.cycle([None, 0.0, None, 1.0, 1.0, None, 3.0, 3.0, 3.0, 3.0])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                next(pool_e),
                random.randint(-10, 10),
                random.randint(-10, 10),
            )
            yield d

    @staticmethod
    def gen_almost_unique_row(N):
        pool = itt.cycle(["a", "b", "c", "d", "e"])
        pool_b = itt.cycle([1.1, 1.2])
        pool_c = itt.cycle([1, 2, 3])
        pool_d = itt.cycle([1, 2, 3])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                random.random(),
                random.randint(-10, 10),
                random.randint(-10, 10),
            )
            yield d

    @staticmethod
    def helper_itt_groupby(data, keyfunc):
        groups = []
        uniquekeys = []
        data = sorted(data, key=keyfunc)
        for k, g in itt.groupby(data, keyfunc):
            groups.append(list(g))
            uniquekeys.append(k)
        result = {"groups": groups, "uniquekeys": uniquekeys}
        return result

    def test_groupby_01(self, engine):
        """
        test_groupby_01: Test groupby's group creation
                         (groupby single row rsults into multiple groups)
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = ["f4", "f5", "f6"]
        num_rows = 2000
        g = self.gen_almost_unique_row(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        # Sort result by groupby columns for consistent comparison
        result_parquery = sort_table(result_parquery, groupby_cols)
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        assert all(
            a == b
            for a, b in zip([x for x in result_parquery["f0"].to_pylist()], uniquekeys)
        )

    def test_groupby_02(self, engine):
        """
        test_groupby_02: Test groupby's group creation
                         (groupby over multiple rows results
                         into multiple groups)
        """
        random.seed(1)
        groupby_cols = ["f0", "f1", "f2"]

        def groupby_lambda(x):
            return [x[0], x[1], x[2]]

        agg_list = ["f4", "f5", "f6"]
        num_rows = 2000
        g = self.gen_almost_unique_row(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = sorted(result_itt["uniquekeys"])
        print(uniquekeys)
        tuple_list = sorted(
            [x["f0"], x["f1"], x["f2"]] for x in result_parquery.to_pylist()
        )
        assert all(a == b for a, b in zip(tuple_list, uniquekeys))

    def test_groupby_03(self, engine):
        """
        test_groupby_03: Test groupby's aggregations
                        (groupby single row results into multiple groups)
                        Groupby type 'sum'
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = ["f4", "f5", "f6"]

        def agg_lambda(x):
            return [x[4], x[5], x[6]]

        num_rows = 2000
        g = self.gen_almost_unique_row(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for item in result_itt["groups"]:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                f4 += row[4]
                f5 += row[5]
                f6 += row[6]
            ref.append([f0, f4, f5, f6])
        result_ref = pa.table(
            {
                col: [row[i] for row in ref]
                for i, col in enumerate(result_parquery.column_names)
            }
        )
        result_ref = round_float_columns(result_ref, 6)
        result_parquery = round_float_columns(result_parquery, 6)
        # Sort for consistent comparison across engines
        result_ref = sort_table(result_ref, groupby_cols)
        result_parquery = sort_table(result_parquery, groupby_cols)
        assert_tables_equal(result_ref, result_parquery, decimal=6)

    def test_groupby_04(self, engine):
        """
        test_groupby_04: Test groupby's aggregation
                             (groupby over multiple rows results
                             into multiple groups)
                             Groupby type 'sum'
        """
        random.seed(1)
        groupby_cols = ["f0", "f1", "f2"]

        def groupby_lambda(x):
            return [x[0], x[1], x[2]]

        agg_list = ["f4", "f5", "f6"]

        def agg_lambda(x):
            return [x[4], x[5], x[6]]

        num_rows = 2000
        g = self.gen_almost_unique_row(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = sorted(result_itt["uniquekeys"])
        print(uniquekeys)
        ref = []
        for item in result_itt["groups"]:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                f4 += row[4]
                f5 += row[5]
                f6 += row[6]
            ref.append(f0 + [f4, f5, f6])
        tuple_list = sorted(
            [x["f0"], x["f1"], x["f2"]] for x in result_parquery.to_pylist()
        )
        assert all(a == b for a, b in zip(tuple_list, uniquekeys))

    def test_groupby_05(self, engine):
        """
        test_groupby_05: Test groupby's group creation without cache
        Groupby type 'sum'
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = ["f1"]
        num_rows = 200
        for _dtype in ["i8", "i4", "f8", "S1"]:
            if _dtype == "S1":
                iterable = ((str(x % 5), x % 5) for x in range(num_rows))
            else:
                iterable = ((x % 5, x % 5) for x in range(num_rows))
            if _dtype == "S1":
                schema = [("f0", "string"), ("f1", "int64")]
            elif _dtype == "i8":
                schema = [("f0", "int64"), ("f1", "int64")]
            elif _dtype == "i4":
                schema = [("f0", "int32"), ("f1", "int64")]
            elif _dtype == "f8":
                schema = [("f0", "double"), ("f1", "int64")]
            table = create_table_from_generator(iterable, schema)
            print("--> ParQuery")
            self.filename = tempfile.mkstemp(prefix="test-")[-1]
            df_to_parquet(table, self.filename)
            result_parquery = aggregate_pq(
                self.filename, groupby_cols, agg_list, as_df=False, engine=engine
            )
            print(result_parquery)
            print("--> Itertools")
            data_list = list(
                zip(*[table[col].to_pylist() for col in table.column_names])
            )
            result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
            uniquekeys = result_itt["uniquekeys"]
            print(uniquekeys)
            ref = []
            for item in result_itt["groups"]:
                f1 = 0
                for row in item:
                    f0 = row[0]
                    f1 += row[1]
                ref.append([f0] + [f1])
            result_ref = pa.table(
                {
                    col: [row[i] for row in ref]
                    for i, col in enumerate(result_parquery.column_names)
                }
            )
            result_ref = round_float_columns(result_ref, 6)
            result_parquery = round_float_columns(result_parquery, 6)
        # Sort for consistent comparison across engines
        result_ref = sort_table(result_ref, groupby_cols)
        result_parquery = sort_table(result_parquery, groupby_cols)
        assert_tables_equal(result_ref, result_parquery, decimal=6)

    def test_groupby_06(self, engine):
        """
        test_groupby_06: Groupby type 'count'
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = [["f4", "count"], ["f5", "count"], ["f6", "count"]]
        num_rows = 2000
        g = self.gen_dataset_count(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for item in result_itt["groups"]:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])
        result_ref = pa.table(
            {
                col: [row[i] for row in ref]
                for i, col in enumerate(result_parquery.column_names)
            }
        )
        result_ref = round_float_columns(result_ref, 6)
        result_parquery = round_float_columns(result_parquery, 6)
        # Sort for consistent comparison across engines
        result_ref = sort_table(result_ref, groupby_cols)
        result_parquery = sort_table(result_parquery, groupby_cols)
        assert_tables_equal(result_ref, result_parquery, decimal=6)

    def test_groupby_07(self, engine):
        """
        test_groupby_07: Groupby type 'count'
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = [["f4", "count"], ["f5", "count"], ["f6", "count"]]
        num_rows = 1000
        g = self.gen_dataset_count_with_NA(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for item in result_itt["groups"]:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                if row[4] is not None:
                    f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])
        result_ref = pa.table(
            {
                col: [row[i] for row in ref]
                for i, col in enumerate(result_parquery.column_names)
            }
        )
        result_ref = round_float_columns(result_ref, 6)
        result_parquery = round_float_columns(result_parquery, 6)
        # Sort for consistent comparison across engines
        result_ref = sort_table(result_ref, groupby_cols)
        result_parquery = sort_table(result_parquery, groupby_cols)
        assert_tables_equal(result_ref, result_parquery, decimal=6)

    def _get_unique(self, values):
        new_values = []
        nan_found = False
        for item in values:
            if item not in new_values:
                if item == item:
                    new_values.append(item)
                elif not nan_found:
                    new_values.append(item)
                    nan_found = True
        return new_values

    def gen_dataset_count_with_NA_08(self, N):
        pool = itt.cycle(["a", "a", "b", "b", "b", "c", "c", "c", "c", "c"])
        pool_b = itt.cycle([0.0, 0.1, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_e = itt.cycle([None, 0.0, None, 0.0, 1.0, None, 3.0, 1.0, 3.0, 1.0])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                next(pool_e),
                random.randint(-500, 500),
                random.randint(-100, 100),
            )
            yield d

    def test_groupby_08(self, engine):
        """
        test_groupby_08: Groupby's type 'count_distinct'
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = [
            ["f4", "count_distinct"],
            ["f5", "count_distinct"],
            ["f6", "count_distinct"],
        ]
        num_rows = 2000
        g = self.gen_dataset_count_with_NA_08(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("table")
        print(table)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for n, (u, item) in enumerate(zip(uniquekeys, result_itt["groups"])):
            f4 = len(self._get_unique([x[4] for x in result_itt["groups"][n]]))
            f5 = len(self._get_unique([x[5] for x in result_itt["groups"][n]]))
            f6 = len(self._get_unique([x[6] for x in result_itt["groups"][n]]))
            ref.append([u, f4, f5, f6])
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = [["f4", "count"], ["f5", "count"], ["f6", "count"]]
        num_rows = 1000
        g = self.gen_dataset_count_with_NA(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for item in result_itt["groups"]:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                if row[4] is not None:
                    f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])
        result_ref = pa.table(
            {
                col: [row[i] for row in ref]
                for i, col in enumerate(result_parquery.column_names)
            }
        )
        result_ref = round_float_columns(result_ref, 6)
        result_parquery = round_float_columns(result_parquery, 6)
        # Sort for consistent comparison across engines
        result_ref = sort_table(result_ref, groupby_cols)
        result_parquery = sort_table(result_parquery, groupby_cols)
        assert_tables_equal(result_ref, result_parquery, decimal=6)

    def test_groupby_09(self, engine):
        """
        test_groupby_09: Groupby's type 'count_distinct' with a large number of records
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = [
            ["f4", "count_distinct"],
            ["f5", "count_distinct"],
            ["f6", "count_distinct"],
        ]
        num_rows = 200000
        g = self.gen_dataset_count_with_NA_08(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("table")
        print(table)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for n, (u, item) in enumerate(zip(uniquekeys, result_itt["groups"])):
            f4 = len(self._get_unique([x[4] for x in result_itt["groups"][n]]))
            f5 = len(self._get_unique([x[5] for x in result_itt["groups"][n]]))
            f6 = len(self._get_unique([x[6] for x in result_itt["groups"][n]]))
            ref.append([u, f4, f5, f6])
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = [["f4", "count"], ["f5", "count"], ["f6", "count"]]
        num_rows = 1000
        g = self.gen_dataset_count_with_NA(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for item in result_itt["groups"]:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                if row[4] is not None:
                    f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])
        result_ref = pa.table(
            {
                col: [row[i] for row in ref]
                for i, col in enumerate(result_parquery.column_names)
            }
        )
        result_ref = round_float_columns(result_ref, 6)
        result_parquery = round_float_columns(result_parquery, 6)
        # Sort for consistent comparison across engines
        result_ref = sort_table(result_ref, groupby_cols)
        result_parquery = sort_table(result_parquery, groupby_cols)
        assert_tables_equal(result_ref, result_parquery, decimal=6)

    def test_groupby_10(self, engine):
        """
        test_groupby_14: Groupby type 'mean'
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = [["f4", "mean"], ["f5", "mean"], ["f6", "mean"]]

        def agg_lambda(x):
            return [x[4], x[5], x[6]]

        num_rows = 2000
        g = self.gen_almost_unique_row(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        # Sort for consistent comparison across engines
        result_parquery = sort_table(result_parquery, groupby_cols)
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for item in result_itt["groups"]:
            f4 = []
            f5 = []
            f6 = []
            for row in item:
                groupby_lambda(row)
                f4.append(row[4])
                f5.append(row[5])
                f6.append(row[6])
            ref.append([statistics.mean(f4), statistics.mean(f5), statistics.mean(f6)])
        result_data = [list(row.values())[1:] for row in result_parquery.to_pylist()]
        for i, (result_row, ref_row) in enumerate(zip(result_data, ref)):
            for j, (r, e) in enumerate(zip(result_row, ref_row)):
                assert math.isclose(r, e, rel_tol=1e-10, abs_tol=1e-10), (
                    f"Row {i}, col {j}: {r} != {e}"
                )

    def test_groupby_11(self, engine):
        """
        test_groupby_11: Groupby type 'std'
        """
        random.seed(1)
        groupby_cols = ["f0"]

        def groupby_lambda(x):
            return x[0]

        agg_list = [["f4", "std"], ["f5", "std"], ["f6", "std"]]

        def agg_lambda(x):
            return [x[4], x[5], x[6]]

        num_rows = 2000
        g = self.gen_almost_unique_row(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        # Sort for consistent comparison across engines
        result_parquery = sort_table(result_parquery, groupby_cols)
        print(result_parquery)
        print("--> Itertools")
        data_list = list(zip(*[table[col].to_pylist() for col in table.column_names]))
        result_itt = self.helper_itt_groupby(data_list, groupby_lambda)
        uniquekeys = result_itt["uniquekeys"]
        print(uniquekeys)
        ref = []
        for item in result_itt["groups"]:
            f4 = []
            f5 = []
            f6 = []
            for row in item:
                groupby_lambda(row)
                f4.append(row[4])
                f5.append(row[5])
                f6.append(row[6])
            ref.append(
                [statistics.stdev(f4), statistics.stdev(f5), statistics.stdev(f6)]
            )
        result_values = [list(x.values())[1:] for x in result_parquery.to_pylist()]
        for i, (result_row, ref_row) in enumerate(zip(result_values, ref)):
            for j, (val, ref_val) in enumerate(zip(result_row, ref_row)):
                if not math.isclose(val, ref_val, rel_tol=0.01, abs_tol=0.01):
                    raise AssertionError(f"Row {i}, col {j}: {val} != {ref_val}")

    def test_groupby_12(self, engine):
        """
        test_groupby_12: Test groupby without groupby column
        """
        random.seed(1)
        groupby_cols = []
        agg_list = ["f4", "f5", "f6"]
        num_rows = 2000
        g = self.gen_almost_unique_row(num_rows)
        schema_7col = [
            ("f0", "string"),
            ("f1", "double"),
            ("f2", "int64"),
            ("f3", "int32"),
            ("f4", "double"),
            ("f5", "int64"),
            ("f6", "int32"),
        ]
        table = create_table_from_generator(g, schema_7col)
        print("--> ParQuery")
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        result_parquery = aggregate_pq(
            self.filename, groupby_cols, agg_list, as_df=False, engine=engine
        )
        print(result_parquery)
        print("--> Numpy")
        np_result = [
            pytest.approx(sum(table["f4"].to_pylist())),
            pytest.approx(sum(table["f5"].to_pylist())),
            pytest.approx(sum(table["f6"].to_pylist())),
        ]
        assert list(result_parquery.to_pylist()[0].values()) == np_result

    def test_where_terms00(self, engine):
        """
        test_where_terms00: get terms in one column bigger than a certain value
        """
        ref = [[x, x] for x in range(10001, 20000)]
        iterable = ((x, x) for x in range(20000))
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", ">", 10000)]
        result_parquery = aggregate_pq(
            self.filename,
            ["f0"],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_where_terms01(self, engine):
        """
        test_where_terms01: get terms in one column less or equal than a
                            certain value
        """
        ref = [[x, x] for x in range(0, 10001)]
        iterable = ((x, x) for x in range(20000))
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", "<=", 10000)]
        result_parquery = aggregate_pq(
            self.filename,
            ["f0"],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_where_terms02(self, engine):
        """
        test_where_terms02: get mask where terms not in list
        """
        exclude = [0, 1, 2, 3, 11, 12, 13]
        ref = [[x, x] for x in range(20000) if x not in exclude]
        iterable = ((x, x) for x in range(20000))
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", "not in", exclude)]
        result_parquery = aggregate_pq(
            self.filename,
            ["f0"],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_where_terms03(self, engine):
        """
        test_where_terms03: get mask where terms in list
        """
        include = [0, 1, 2, 3, 11, 12, 13]
        ref = [[x, x] for x in range(20000) if x in include]
        iterable = ((x, x) for x in range(20000))
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", "in", include)]
        result_parquery = aggregate_pq(
            self.filename,
            ["f0"],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_where_terms_04(self, engine):
        """
        test_where_terms04: get mask where terms in list with only one item
        """
        include = [0]
        ref = [[x, x] for x in range(20000) if x in include]
        iterable = ((x, x) for x in range(20000))
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", "in", include)]
        result_parquery = aggregate_pq(
            self.filename,
            ["f0"],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_where_terms_05(self, engine):
        """
        test_where_terms05: get mask where terms in list with a filter on unused column without aggregation
        """
        include = [0, 1000, 2000]
        ref = [[x] for x in range(20000) if x in include]
        iterable = ((x, x) for x in range(20000))
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", "in", include)]
        result_parquery = aggregate_pq(
            self.filename,
            [],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_where_terms_06(self, engine):
        """
        test_where_terms06: get mask where terms in list with a filter on unused column with aggregation
        """
        include = [0, 1000, 2000]
        ref = [[x] for x in range(20000) if x == 0 + 1000 + 2000]
        iterable = ((x, x) for x in range(20000))
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", "in", include)]
        result_parquery = aggregate_pq(
            self.filename,
            [],
            ["f1"],
            data_filter=terms_filter,
            aggregate=True,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_where_terms07(self, engine):
        """
        test_where_terms07: get mask on where terms in a very long list
        """
        include = [0, 1, 2, 3, 11, 12, 13]
        ref = [[x, x] for x in range(20000) if x in include]
        include *= 100
        iterable = ((x, x) for x in range(20000))
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", "in", include)]
        result_parquery = aggregate_pq(
            self.filename,
            ["f0"],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_natural_notation(self, engine):
        """
        test_natural_notation: check the handling of difficult naming
        """
        include = [0, 1000, 2000]
        ref = [[x, x] for x in range(20000) if x in [0, 1000, 2000]]
        iterable = ((x, x, x) for x in range(20000))
        schema_3col = [("f0", "int64"), ("f1", "int64"), ("f2", "int64")]
        table = create_table_from_generator(iterable, schema_3col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("d-1", "in", include)]
        result_parquery = aggregate_pq(
            self.filename,
            ["d-2"],
            ["m-1"],
            data_filter=terms_filter,
            aggregate=True,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_natural_notation_2(self, engine):
        """
        test_natural_notation: check the handling of difficult naming without dimensions
        """
        include = [0, 1000, 2000]
        ref = [[x] for x in range(20000) if x == 0 + 1000 + 2000]
        iterable = ((x, x, x) for x in range(20000))
        schema_3col = [("f0", "int64"), ("f1", "int64"), ("f2", "int64")]
        table = create_table_from_generator(iterable, schema_3col)
        filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, filename)
        terms_filter = [("d-1", "in", include)]
        result_parquery = aggregate_pq(
            filename,
            [],
            ["m-1"],
            data_filter=terms_filter,
            aggregate=True,
            as_df=False,
            engine=engine,
        )
        assert all(
            a == b
            for a, b in zip(
                [list(row.values()) for row in result_parquery.to_pylist()], ref
            )
        )

    def test_non_existing_column(self, engine):
        """
        test_non_existing_column: check the handling of missing columns in the parquet file
        measure columns should get 0.0 as value
        dimension columns should get the default -1 (unknown) identifier
        """
        iterable = ((x, x, x) for x in range(20000))
        schema_3col = [("f0", "int64"), ("f1", "int64"), ("f2", "int64")]
        table = create_table_from_generator(iterable, schema_3col)
        filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, filename)
        result_parquery = aggregate_pq(
            filename,
            ["d1", "d3"],
            ["m1", "m2"],
            data_filter=[],
            aggregate=True,
            as_df=False,
            engine=engine,
        )
        assert sum(result_parquery["m2"].to_pylist()) == 0.0
        assert list(set(result_parquery["d3"].to_pylist())) == [-1]

    def test_non_existing_file(self, engine):
        """
        test_non_existing_file: check the handling of missing files
        measure columns should get 0.0 as value
        dimension columns should get the default -1 (unknown) identifier
        """
        result_parquery = aggregate_pq(
            "not_existing_file.$$$",
            ["d1", "d3"],
            ["m1", "m2"],
            data_filter=[],
            aggregate=True,
            as_df=False,
            engine=engine,
        )
        assert result_parquery.num_rows == 0
        assert sorted(result_parquery.column_names) == ["d1", "d3", "m1", "m2"]

    def test_emtpy_file(self, engine):
        """
        When a file is empty (does not contain any rows) it should still work.
        """
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        data_table = pa.table({"f0": [], "f1": []})
        with pa.parquet.ParquetWriter(
            self.filename, data_table.schema, version="2.6", compression="ZSTD"
        ) as writer:
            writer.write_table(data_table)
        terms_filter = [("f0", ">", 10000)]
        result_parquery = aggregate_pq(
            self.filename,
            ["f0"],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert result_parquery.num_rows == 0
        assert set(result_parquery.column_names) == {"f0", "f1"}

    def test_all_results_filtered(self, engine):
        """
        test_where_terms00: get terms in one column bigger than a certain value
        """
        iterable = ((x * mult, x * mult) for x in range(5000) for mult in [1, 3])
        schema_2col = [("f0", "int64"), ("f1", "int64")]
        table = create_table_from_generator(iterable, schema_2col)
        self.filename = tempfile.mkstemp(prefix="test-")[-1]
        df_to_parquet(table, self.filename)
        terms_filter = [("f0", "in", [8000, 13000])]
        result_parquery = aggregate_pq(
            self.filename,
            ["f0"],
            ["f1"],
            data_filter=terms_filter,
            aggregate=False,
            as_df=False,
            engine=engine,
        )
        assert result_parquery.num_rows == 0
