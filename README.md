# parq-tools
[![Run Tests](https://github.com/Elphick/parq-tools/actions/workflows/build_and_test.yml/badge.svg?branch=main)](https://github.com/Elphick/parq-tools/actions/workflows/build_and_test.yml)
[![PyPI](https://img.shields.io/pypi/v/parq-tools.svg?logo=python&logoColor=white)](https://pypi.org/project/parq-tools/)
![Coverage](https://raw.githubusercontent.com/elphick/parq-tools/main/docs/source/_static/badges/coverage.svg)
[![Python Versions](https://img.shields.io/pypi/pyversions/parq-tools.svg)](https://pypi.org/project/parq-tools/)
[![License](https://img.shields.io/github/license/Elphick/parq-tools.svg?logo=apache&logoColor=white)](https://pypi.org/project/parq-tools/)
[![Publish Docs](https://github.com/Elphick/parq-tools/actions/workflows/docs_to_gh_pages.yml/badge.svg?branch=main)](https://github.com/Elphick/parq-tools/actions/workflows/docs_to_gh_pages.yml)
[![Open Issues](https://img.shields.io/github/issues/Elphick/parq-tools.svg)](https://github.com/Elphick/parq-tools/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/Elphick/parq-tools.svg)](https://github.com/Elphick/parq-tools/pulls)


## Overview
`parq-tools` is a collection of utilities for efficiently working with **large-scale** Parquet datasets.
A typical use case is asset-based workflows with large scientific datasets.

:::note
If your datasets are not large, you might find the `pandas` library more convenient.
:::

## Features
- [x] **Filtering** → Efficiently filter large parquet files.
- [x] **Concatenation** → Combines multiple Parquet files efficiently along rows (`axis=0`) or columns (`axis=1`).
- [x] **Tokenized Filtering** → Converts **pandas-style expressions** into efficient PyArrow queries.
- [x] **Profiling Enhancements** → Improves `ydata-profiling` by profiling **specific columns incrementally**, merging results for large files.
- [ ] **DataFrame Enhancements** → Provides a `LazyParquetDataFrame` class that extends `pandas.DataFrame` with lazy loading from Parquet files.