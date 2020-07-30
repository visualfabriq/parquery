import itertools as itt
import os
import random
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from parquery import df_to_parquet, aggregate_pq


class TestParquery(object):
    @contextmanager
    def on_disk_data_cleaner(self, data):
        # write
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        yield self.filename

        shutil.rmtree(self.rootdir)
        self.rootdir = None

    def setup(self):
        self.filename = None

    def teardown(self):
        if self.filename:
            os.remove(self.filename)
            self.filename = None

    def gen_dataset_count(self, N):
        pool = itt.cycle(['a', 'a',
                          'b', 'b', 'b',
                          'c', 'c', 'c', 'c', 'c'])
        pool_b = itt.cycle([0.0, 0.0,
                            1.0, 1.0, 1.0,
                            3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                random.random(),
                random.randint(- 10, 10),
                random.randint(- 10, 10),
            )
            yield d

    def gen_dataset_count_with_NA(self, N):
        pool = itt.cycle(['a', 'a',
                          'b', 'b', 'b',
                          'c', 'c', 'c', 'c', 'c'])
        pool_b = itt.cycle([0.0, 0.1,
                            1.0, 1.0, 1.0,
                            3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_e = itt.cycle([np.nan, 0.0,
                            np.nan, 1.0, 1.0,
                            np.nan, 3.0, 3.0, 3.0, 3.0])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                next(pool_e),
                random.randint(- 10, 10),
                random.randint(- 10, 10),
            )
            yield d

    def gen_almost_unique_row(self, N):
        pool = itt.cycle(['a', 'b', 'c', 'd', 'e'])
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
                random.randint(- 10, 10),
                random.randint(- 10, 10),
            )
            yield d

    def helper_itt_groupby(self, data, keyfunc):
        groups = []
        uniquekeys = []
        data = sorted(data,
                      key=keyfunc)  # mandatory before calling itertools groupby!
        for k, g in itt.groupby(data, keyfunc):
            groups.append(list(g))  # Store group iterator as a list
            uniquekeys.append(k)

        result = {
            'groups': groups,
            'uniquekeys': uniquekeys
        }
        return result

    def test_groupby_01(self):
        """
        test_groupby_01: Test groupby's group creation
                         (groupby single row rsults into multiple groups)
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        # no operation is specified in `agg_list`, so `sum` is used by default.
        agg_list = ['f4', 'f5', 'f6']
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]

        df_to_parquet(pd.DataFrame(data), self.filename)
        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        assert all(a == b for a, b in zip([x for x in result_parquery['f0']], uniquekeys))

    def test_groupby_02(self):
        """
        test_groupby_02: Test groupby's group creation
                         (groupby over multiple rows results
                         into multiple groups)
        """
        random.seed(1)

        groupby_cols = ['f0', 'f1', 'f2']
        groupby_lambda = lambda x: [x[0], x[1], x[2]]
        # no operation is specified in `agg_list`, so `sum` is used by default.
        agg_list = ['f4', 'f5', 'f6']
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]

        df_to_parquet(pd.DataFrame(data), self.filename)
        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        assert all(
            a == b for a, b in zip([[x['f0'], x['f1'], x['f2']] for _, x in result_parquery.iterrows()],
                                   uniquekeys))

    def test_groupby_03(self):
        """
        test_groupby_03: Test groupby's aggregations
                        (groupby single row results into multiple groups)
                        Groupby type 'sum'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = ['f4', 'f5', 'f6']
        agg_lambda = lambda x: [x[4], x[5], x[6]]
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]

        df_to_parquet(pd.DataFrame(data), self.filename)
        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                f4 += row[4]
                f5 += row[5]
                f6 += row[6]
            ref.append([f0, f4, f5, f6])

        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_groupby_04(self):
        """
        test_groupby_04: Test groupby's aggregation
                             (groupby over multiple rows results
                             into multiple groups)
                             Groupby type 'sum'
        """
        random.seed(1)

        groupby_cols = ['f0', 'f1', 'f2']
        groupby_lambda = lambda x: [x[0], x[1], x[2]]
        agg_list = ['f4', 'f5', 'f6']
        agg_lambda = lambda x: [x[4], x[5], x[6]]
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]

        df_to_parquet(pd.DataFrame(data), self.filename)
        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                f4 += row[4]
                f5 += row[5]
                f6 += row[6]
            ref.append(f0 + [f4, f5, f6])

        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_groupby_05(self):
        """
        test_groupby_05: Test groupby's group creation without cache
        Groupby type 'sum'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = ['f1']
        num_rows = 200

        for _dtype in \
                [
                    'i8',
                    'i4',
                    'f8',
                    'S1',
                ]:

            # -- Data --
            if _dtype == 'S1':
                iterable = ((str(x % 5), x % 5) for x in range(num_rows))
            else:
                iterable = ((x % 5, x % 5) for x in range(num_rows))

            data = np.fromiter(iterable, dtype=_dtype + ',i8')

            # -- ParQuery --
            print('--> ParQuery')
            self.filename = tempfile.mkstemp(prefix='test-')[-1]
            df_to_parquet(pd.DataFrame(data), self.filename)

            result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
            print(result_parquery)

            # Itertools result
            print('--> Itertools')
            result_itt = self.helper_itt_groupby(data, groupby_lambda)
            uniquekeys = result_itt['uniquekeys']
            print(uniquekeys)

            ref = []
            for item in result_itt['groups']:
                f1 = 0
                for row in item:
                    f0 = row[0]
                    f1 += row[1]
                ref.append([f0] + [f1])

            assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_groupby_06(self):
        """
        test_groupby_06: Groupby type 'count'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count'], ['f5', 'count'], ['f6', 'count']]
        num_rows = 2000

        # -- Data --
        g = self.gen_dataset_count(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])

        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_groupby_07(self):
        """
        test_groupby_07: Groupby type 'count'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count'], ['f5', 'count'], ['f6', 'count']]
        num_rows = 1000

        # -- Data --
        g = self.gen_dataset_count_with_NA(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                if row[4] == row[4]:
                    f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])

        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def _get_unique(self, values):
        new_values = []
        nan_found = False

        for item in values:
            if item not in new_values:
                if item == item:
                    new_values.append(item)
                else:
                    if not nan_found:
                        new_values.append(item)
                        nan_found = True

        return new_values

    def gen_dataset_count_with_NA_08(self, N):
        pool = itt.cycle(['a', 'a',
                          'b', 'b', 'b',
                          'c', 'c', 'c', 'c', 'c'])
        pool_b = itt.cycle([0.0, 0.1,
                            1.0, 1.0, 1.0,
                            3.0, 3.0, 3.0, 3.0, 3.0])
        pool_c = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_d = itt.cycle([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        pool_e = itt.cycle([np.nan, 0.0,
                            np.nan, 0.0, 1.0,
                            np.nan, 3.0, 1.0, 3.0, 1.0])
        for _ in range(N):
            d = (
                next(pool),
                next(pool_b),
                next(pool_c),
                next(pool_d),
                next(pool_e),
                random.randint(- 500, 500),
                random.randint(- 100, 100),
            )
            yield d

    def test_groupby_08(self):
        """
        test_groupby_08: Groupby's type 'count_distinct'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count_distinct'], ['f5', 'count_distinct'], ['f6', 'count_distinct']]
        num_rows = 2000

        # -- Data --
        g = self.gen_dataset_count_with_NA_08(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')
        print('data')
        print(data)

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)
        #
        # # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []

        for n, (u, item) in enumerate(zip(uniquekeys, result_itt['groups'])):
            f4 = len(self._get_unique([x[4] for x in result_itt['groups'][n]]))
            f5 = len(self._get_unique([x[5] for x in result_itt['groups'][n]]))
            f6 = len(self._get_unique([x[6] for x in result_itt['groups'][n]]))
            ref.append([u, f4, f5, f6])

        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count'], ['f5', 'count'], ['f6', 'count']]
        num_rows = 1000

        # -- Data --
        g = self.gen_dataset_count_with_NA(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                if row[4] == row[4]:
                    f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])

        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_groupby_09(self):
        """
        test_groupby_09: Groupby's type 'count_distinct' with a large number of records
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count_distinct'], ['f5', 'count_distinct'], ['f6', 'count_distinct']]
        num_rows = 200000

        # -- Data --
        g = self.gen_dataset_count_with_NA_08(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')
        print('data')
        print(data)

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)
        #
        # # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []

        for n, (u, item) in enumerate(zip(uniquekeys, result_itt['groups'])):
            f4 = len(self._get_unique([x[4] for x in result_itt['groups'][n]]))
            f5 = len(self._get_unique([x[5] for x in result_itt['groups'][n]]))
            f6 = len(self._get_unique([x[6] for x in result_itt['groups'][n]]))
            ref.append([u, f4, f5, f6])

        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'count'], ['f5', 'count'], ['f6', 'count']]
        num_rows = 1000

        # -- Data --
        g = self.gen_dataset_count_with_NA(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = 0
            f5 = 0
            f6 = 0
            for row in item:
                f0 = groupby_lambda(row)
                if row[4] == row[4]:
                    f4 += 1
                f5 += 1
                f6 += 1
            ref.append([f0, f4, f5, f6])

        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_groupby_10(self):
        """
        test_groupby_14: Groupby type 'mean'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'mean'], ['f5', 'mean'], ['f6', 'mean']]
        agg_lambda = lambda x: [x[4], x[5], x[6]]
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = []
            f5 = []
            f6 = []
            for row in item:
                f0 = groupby_lambda(row)
                f4.append(row[4])
                f5.append(row[5])
                f6.append(row[6])

            ref.append([np.mean(f4), np.mean(f5), np.mean(f6)])

        # remove the first (text) element for floating point comparison
        assert_allclose([list(x)[1:] for x in result_parquery.to_numpy()], ref, rtol=1e-10)

    def test_groupby_11(self):
        """
        test_groupby_11: Groupby type 'std'
        """
        random.seed(1)

        groupby_cols = ['f0']
        groupby_lambda = lambda x: x[0]
        agg_list = [['f4', 'std'], ['f5', 'std'], ['f6', 'std']]
        agg_lambda = lambda x: [x[4], x[5], x[6]]
        num_rows = 2000

        # -- Data --
        g = self.gen_almost_unique_row(num_rows)
        data = np.fromiter(g, dtype='S1,f8,i8,i4,f8,i8,i4')

        # -- ParQuery --
        print('--> ParQuery')
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        result_parquery = aggregate_pq(self.filename, groupby_cols, agg_list)
        print(result_parquery)

        # Itertools result
        print('--> Itertools')
        result_itt = self.helper_itt_groupby(data, groupby_lambda)
        uniquekeys = result_itt['uniquekeys']
        print(uniquekeys)

        ref = []
        for item in result_itt['groups']:
            f4 = []
            f5 = []
            f6 = []
            for row in item:
                f0 = groupby_lambda(row)
                f4.append(row[4])
                f5.append(row[5])
                f6.append(row[6])

            ref.append([np.std(f4), np.std(f5), np.std(f6)])

        # remove the first (text) element for floating point comparison
        # nb: here we do a less specific check to make sureit passes
        assert_allclose([list(x)[1:] for x in result_parquery.to_numpy()], ref, rtol=1e-2)

    def test_where_terms00(self):
        """
        test_where_terms00: get terms in one column bigger than a certain value
        """
        # expected result
        ref = [[x, x] for x in range(10001, 20000)]

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        # filter data
        terms_filter = [('f0', '>', 10000)]
        result_parquery = aggregate_pq(self.filename, ['f0'], ['f1'],
                                       data_filter=terms_filter,
                                       aggregate=False)

        # compare
        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_where_terms01(self):
        """
        test_where_terms01: get terms in one column less or equal than a
                            certain value
        """
        # expected result
        ref = [[x, x] for x in range(0, 10001)]

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        # filter data
        terms_filter = [('f0', '<=', 10000)]
        result_parquery = aggregate_pq(self.filename, ['f0'], ['f1'],
                                       data_filter=terms_filter,
                                       aggregate=False)

        # compare
        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_where_terms02(self):
        """
        test_where_terms02: get mask where terms not in list
        """
        exclude = [0, 1, 2, 3, 11, 12, 13]

        # expected result
        ref = [[x, x] for x in range(20000) if x not in exclude]

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        # filter data
        terms_filter = [('f0', 'not in', exclude)]
        result_parquery = aggregate_pq(self.filename, ['f0'], ['f1'],
                                       data_filter=terms_filter,
                                       aggregate=False)

        # compare
        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_where_terms03(self):
        """
        test_where_terms03: get mask where terms in list
        """
        include = [0, 1, 2, 3, 11, 12, 13]

        # expected result
        ref = [[x, x] for x in range(20000) if x in include]

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        # filter data
        terms_filter = [('f0', 'in', include)]
        result_parquery = aggregate_pq(self.filename, ['f0'], ['f1'],
                                       data_filter=terms_filter,
                                       aggregate=False)

        # compare
        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))

    def test_where_terms_04(self):
        """
        test_where_terms04: get mask where terms in list with only one item
        """

        include = [0]

        # expected result
        ref = [[x, x] for x in range(20000) if x in include]

        # generate data to filter on
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')

        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        # filter data
        terms_filter = [('f0', 'in', include)]
        result_parquery = aggregate_pq(self.filename, ['f0'], ['f1'],
                                       data_filter=terms_filter,
                                       aggregate=False)

        # compare
        assert all(a == b for a, b in zip([list(x) for x in result_parquery.to_numpy()], ref))
