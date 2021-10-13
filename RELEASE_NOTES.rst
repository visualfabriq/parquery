========================
Release notes for parquery
========================

Release  0.2.1
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
