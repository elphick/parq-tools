# parq-tools
[![License](https://img.shields.io/github/license/Elphick/parq-tools.svg?logo=apache&logoColor=white)](https://pypi.org/project/parq-tools/)
[![PyPI](https://img.shields.io/pypi/v/parq-tools.svg?logo=python&logoColor=white)](https://pypi.org/project/parq-tools/)
[![Run Tests](https://github.com/Elphick/parq-tools/actions/workflows/poetry_build_and_test.yml/badge.svg?branch=main)](https://github.com/Elphick/parq-tools/actions/workflows/poetry_build_and_test.yml)
[![Publish Docs](https://github.com/Elphick/parq-tools/actions/workflows/poetry_sphinx_docs_to_gh_pages.yml/badge.svg?branch=main)](https://github.com/Elphick/parq-tools/actions/workflows/poetry_sphinx_docs_to_gh_pages.yml)

## Overview
`parq-tools` is a collection of utilities for efficiently working with **large-scale Parquet datasets**. Designed for **scalability**, it supports **chunk-wise processing**, **metadata handling**, and **optimized workflows** for datasets too large to fit into memory.

## Features
- [x] **Filtering** → Efficiently filter large parquet files.
- [x] **Concatenation** → Combines multiple Parquet files efficiently along rows (`axis=0`) or columns (`axis=1`).
- [x] **Tokenized Filtering** → Converts **pandas-style expressions** into efficient PyArrow queries.
- [ ] **Block Model Generation** → Creates **massive Parquet datasets** that exceed memory limits, useful for testing pipelines.
- [ ] **Profiling Enhancements** → Improves `ydata-profiling` by profiling **specific columns incrementally**, merging results for large files.
