########################################################################
#       File based on https://github.com/Blosc/bcolz
########################################################################
#
# License: BSD
# Created: October 5, 2015
#       Author:  Carst Vaartjes - cvaartjes@visualfabriq.com
#
########################################################################
from __future__ import absolute_import

import codecs
import os
from sys import version_info as v

from setuptools import find_packages, setup

# Check this Python version is supported
if v < (3, 7):
    raise Exception("Unsupported Python version %d.%d. Requires Python >= 3.7" % v[:2])

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


# Sources & libraries
sources = []
optional_libs = []
install_requires = [
    "pyarrow>=22.0.0",
]
setup_requires = []
tests_requires = [
    "pytest",
    "coverage",
]
extras_requires = [
    "numpy",
    'pandas<=1.1.5;python_version<"3.11"',
    'pandas>=1.5.3;python_version>="3.11"',
]
extensions = []

package_data = {}
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
]

setup(
    name="parquery",
    description="A query and aggregation framework for Parquet",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=classifiers,
    author="Carst Vaartjes",
    author_email="cvaartjes@visualfabriq.com",
    maintainer="Jelle Verstraaten",
    maintainer_email="jverstraaten@visualfabriq.com",
    url="https://github.com/visualfabriq/parquery",
    license="MIT",
    platforms=["any"],
    ext_modules=extensions,
    cmdclass={},
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_requires,
    extras_require=dict(optional=extras_requires, test=tests_requires),
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    zip_safe=True,
)
