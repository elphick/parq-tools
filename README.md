# parq-tools 🚀

## Overview
`parq-tools` is a collection of utilities for efficiently working with **large-scale Parquet datasets**. Designed for **scalability**, it supports **chunk-wise processing**, **metadata handling**, and **optimized workflows** for datasets too large to fit into memory.

## Features
- [x] **Filtering** → Efficiently filter large parquet files.
- [x] **Concatenation (`ParquetConcat`)** → Combines multiple Parquet files efficiently along rows (`axis=0`) or columns (`axis=1`).
- [ ] **Block Model Generation** → Creates **massive Parquet datasets** that exceed memory limits, useful for testing pipelines.
- [ ] **Profiling Enhancements** → Improves `ydata-profiling` by profiling **specific columns incrementally**, merging results for large files.
- [ ] **Tokenized Filtering** → Converts **pandas-style expressions** into efficient PyArrow queries.
