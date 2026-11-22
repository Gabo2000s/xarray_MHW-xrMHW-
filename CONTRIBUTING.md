# Contributing to xrMHW

Thank you for your interest in contributing to `xrMHW`! We welcome contributions from oceanographers, climate scientists, software engineers, and anyone interested in marine heatwave analysis.

As a scientific package, maintaining numerical reproducibility and computational efficiency is our priority.

## Getting help

* **Questions about usage** (how to chunk a dataset, which baseline to choose, how to read
  an output): open an issue on the
  [issue tracker](https://github.com/Gabo2000s/xarray_MHW-xrMHW-/issues) with the label
  `question`. Include the code you ran and, if possible, a small dataset or the output of
  `print(ds)`.
* **Start with the examples:** the notebooks in [`examples/`](examples/) cover a single
  time series, sea-surface and depth-resolved grids, many files, baselines and
  troubleshooting (`05_troubleshooting.ipynb`).

## Support and maintenance

xrMHW is maintained by G. S. Gutiérrez-Cárdenas as part of research at the Fisheries and
Marine Institute of Memorial University of Newfoundland. Issues and pull requests are
answered on a best-effort basis, normally within two weeks. Bug reports that affect
numerical results take priority. Releases are tagged on GitHub and archived on Zenodo;
changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Code of conduct

Everyone taking part in the project is expected to follow the
[code of conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### 1. Reporting Issues
If you encounter a bug, discrepancy in numerical results, or have a feature request:
1. Search the existing issues to ensure it has not been reported yet.
2. Open a new issue with a clear title and description.
3. If reporting a bug, please include:
   * A minimal working example (MWE) to reproduce the bug.
   * The version of `xrMHW` and key dependencies (`xarray`, `dask`, `numpy`, `pandas`).
   * The expected output vs. the actual output.

### 2. Submitting Pull Requests
We welcome pull requests for bug fixes, performance optimizations, and documentation improvements.
1. Fork the repository and create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Set up the development environment and install dependencies in editable mode with test dependencies:
   ```bash
   pip install -e ".[test]"
   ```
3. Implement your changes. Make sure to document new functions and parameters following the NumPy docstring standard.
4. Add unit tests for any new logic in the `tests/` directory.
5. Run the existing test suite to ensure no regressions were introduced:
   ```bash
   pytest
   ```
6. Commit your changes with clear, descriptive commit messages.
7. Push your branch to GitHub and open a Pull Request against the `main` branch.

## Code Style & Standards

* **Docstrings:** All public functions, classes, and methods should have docstrings using the NumPy docstring format.
* **Vectorization:** Avoid using Python `for` loops across spatial coordinates (latitude, longitude, depth). Use `xarray.apply_ufunc` and Dask-compatible NumPy vectorization to maintain parallel performance.
* **Leap Years:** Ensure that seasonal climatology calculations handle February 29 appropriately.

## License
By contributing to `xrMHW`, you agree that your contributions will be licensed under the GPL-3.0 license.
