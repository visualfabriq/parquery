========================
Release notes for parquery
========================

Release  0.4.0
==============
- Unpinned pyarrow since we have a newer version now
- Implemented python 3 (pyarrow 9+) specific pyarrow logic to optimize memory consumption

Release  0.3.4
==============
- Pinned pyarrow version because the new 9.0.0 causes segfaults

Release  0.3.3
==============
- Skip row_group if output is empty
- Fixes bug that min and max can't be calculated in an empty row group
- Refactored rowgroup_metadata_filter

Release  0.3.2
==============
- Add handling of missing columns in a parquet file that is used in a filter. This happens when new dimensions are created but existing parquet files do not have them yet. Now it throws an error for the query, the new behaviour will change this to giving an empty result. This is better because as the real value for the dimension is unknown for the file, the result should also be zero. It also greatly helps with issues where old files break reporting because they have not been updated yet.
- Removed specification of parquet format (2.0 is now old)

Release  0.2.7, 0.2.8
==============
- Fix an import issue

Release  0.2.6
==============
- Add a parameter to handle default response when a parquet file is missing

Release  0.2.5
==============
- Handle the request for non-existing columns in a parquet file

Release  0.2.1, 0.2.2, 0.2.3, 0.2.4
==============
- Align requirements over requirements.txt, pyproyect.toml and setup.py

Release  0.2.0
==============
- Improve performance for complex list filters
- Improve performance by only aggregating end result (at the cost of some memory efficiency)
- Fixed circleci python 3 tests (rounding and pip versioning)

Release  0.1.16
==============
- Handle count by aggregated results

Release  0.1.12-15
==============
- Enforce order of columns for partial results

Release  0.1.11
==============
- Handle non-natural naming ("-" in column names)

Release  0.1.10
==============
- Check for filter columns that are not part of the result

Release  0.1.9
==============
- Remove the entire uses of categorical values as they impede concatenation of results

Release  0.1.8
==============
- Ensure that groupby columns are seen as categorical series

Release  0.1.7
==============
- Fix Python 2 legacy differences in pyarrow

Release  0.1.6
==============
- Fix Python 2 requirements

Release  0.1.5
==============
- Updated Links

Release  0.1.4
==============
- Added arrow aggregation method

Release  0.1.3
==============
- Introduced writer debug output

Release  0.1.2
==============
- Updated manifest

Release  0.1.1
==============
- Updated requirements for dependencies based on the python version

Release  0.1.1
==============
- Inital release

.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
