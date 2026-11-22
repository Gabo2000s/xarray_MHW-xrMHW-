# Changelog

All notable changes to xrMHW are listed here. Versions follow
[semantic versioning](https://semver.org/). Each release is archived on Zenodo under the
concept DOI [10.5281/zenodo.18343767](https://doi.org/10.5281/zenodo.18343767).

## [3.1.0] — unreleased

### Added
- `xrMHW_clim` computes the climatology and threshold on their own; `xrMHW_events`
  applies a precomputed threshold (from `xrMHW_clim` or built by hand, on a `doy` or
  `time` axis) to a temperature field. `xrMHW_func` still runs both in one pass.
- Five baselines (Smith et al., 2025) through `baseline=`: `"fixed"` (default, as
  before), `"periodic"`, `"detrended"` (linear or polynomial), `"shifting"` (lagged or
  centred, optionally detrended first) and `"adaptation"`.
- `xrMHW_detrend` (per-cell trend removal) and `xrMHW_doy` (day of year on the 366-day
  grid used by xrMHW).
- `xrMHW_annual` and `xrMHW_monthly`: per-cell summaries per year (or block of years)
  and per month, following marineHeatWaves' `blockAverage`, with two conventions for
  cumulative intensity.
- `xrMHW_trend`: per-cell Theil–Sen or least-squares trends with confidence intervals,
  Mann–Kendall (optionally Hamed–Rao corrected) or t tests, and false-discovery-rate
  control over the field or per depth level.
- Onset and decline rates as optional daily outputs.
- Example notebook `06_baselines.ipynb`.

### Changed
- A time axis decoded with cftime in the standard calendar is now accepted; any other
  calendar (`noleap`, `360_day`, `julian`, …) raises a clear `ValueError`. Previously a
  `noleap` axis returned no events without an error.
- Python 3.10–3.13 (Python 3.9 is past end of life). CI runs on every pull request.

### Unchanged
- `xrMHW_func` with the fixed baseline, `detect_mhw_core` and `mhw_1d_wrapper` give the
  same results as in 3.0.0.

## [3.0.0] — unreleased

### Added
- Severity index `{prefix}_severity`, returned by default.
- `output_vars` to select the returned fields.
- Packaging with `pyproject.toml`; `test` and `examples` extras.
- Test suite in `tests/`, validated against a vendored copy of marineHeatWaves; CI on
  GitHub Actions, including a clean install from the built wheel.
- Six example notebooks with data downloaded from Zenodo, benchmark scripts, results
  and the performance figure.
- A time axis split over several Dask chunks is rechunked with a warning instead of
  failing.

### Changed (results may differ from 2.x)
- `threshold` is renamed `{prefix}_threshold`; `detect_mhw_core` and `mhw_1d_wrapper`
  return nine arrays (severity last).
- The climatology pools consecutive calendar days around each occurrence, as the
  reference implementation does.
- Categories are evaluated at the severity peak and capped at 4 (Extreme).
- Days with an undefined threshold no longer count as exceedances.

See "Upgrading from v2.x" in the README.

## [2.0.0] — 2026-02-04

### Added
- `join_method` (`"post-filter"`, default, or `"pre-filter"`) to choose whether short
  gaps are joined after or before the minimum-duration filter.

### Changed
- The climatology smoothing wraps around the year (circular running mean).
- License changed to GPL-3.0.

## [1.0.0] — 2026-01-22

First release: detection of marine heatwaves and cold spells following Hobday et al.
(2016) on xarray datasets of any dimensionality, parallelised with Dask
(`xrMHW_func`, `mhw_1d_wrapper`, `detect_mhw_core`).
