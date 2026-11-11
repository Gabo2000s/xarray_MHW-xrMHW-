# xrMHW — Multidimensional Marine Heatwave Detector (Xarray + Dask)

A parallel, out-of-core Python implementation of the marine heatwave (MHW) definition of
[Hobday et al. (2016)](https://doi.org/10.1016/j.pocean.2015.12.014), applied natively to
N-dimensional ocean grids (time × latitude × longitude × **depth**).

[![Tests](https://github.com/Gabo2000s/xarray_MHW-xrMHW-/actions/workflows/tests.yml/badge.svg)](https://github.com/Gabo2000s/xarray_MHW-xrMHW-/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18476144.svg)](https://doi.org/10.5281/zenodo.18476144)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-GPL--3.0-green)

Developed by Gabriel S. Gutiérrez-Cárdenas ([ORCID 0000-0002-3915-7684](https://orcid.org/0000-0002-3915-7684)).
The detection logic is adapted from the reference implementation
[`marineHeatWaves`](https://github.com/ecjoliver/marineHeatWaves) by Eric C. J. Oliver.

---

## What xrMHW does, and why

Marine heatwaves are increasingly studied below the surface, where they can be more
intense, longer-lived and ecologically more consequential than their surface expression
(Sun et al., 2023; Malan et al., 2025). Detecting them in a depth-resolved reanalysis
means applying the Hobday definition independently to every cell of a four-dimensional
grid — for a domain such as the Grand Banks at 1/12° with 33 depth levels and three
decades of daily fields, that is hundreds of thousands of time series and far more data
than fits in memory.

The established tools were not built for that shape of problem. Oliver's
`marineHeatWaves` and Schlegel & Smit's `heatwaveR` operate on a single time series at a
time; `xmhw` brings the definition to xarray but targets sea surface temperature.
xrMHW keeps the definition intact and changes the execution model: detection is mapped
over the spatial dimensions with `xarray.apply_ufunc` and Dask, so the computation
streams over chunks, uses every available core, and returns results on the input grid
with depth carried through untouched. It adds the diagnostics that depth-resolved work
needs — in particular a dimensionless **severity index**, comparable across depths, and
marine cold spell detection — while reproducing the reference implementation to
floating-point precision in its default configuration (see
[Validation](#validation-against-the-reference-implementation)).

## Scope and limitations

**xrMHW computes pointwise fields.** Every grid cell is treated as an independent 1-D
time series, exactly as in the reference implementation. This has a consequence worth
stating plainly: `mhw_duration` is the persistence of the exceedance **in that
individual cell**, not the lifetime of a coherent three-dimensional structure. On a 4-D
grid, neighbouring cells may report different durations for what is physically one
event. Connected-component labelling, object tracking and event-level (volumetric)
metrics are deliberately out of scope.

Other limitations:

- **Daily resolution.** The climatology windows and the minimum-duration filter are
  defined in days.
- **Standard calendars only.** The time coordinate must be convertible with
  `.toordinal()`; `cftime` calendars (`noleap`, `360_day`) raise a `TypeError`. Convert
  the calendar before calling xrMHW.
- **No spatial aggregation.** Regional averages, event catalogues and composites are
  left to the user's downstream analysis.

## Upgrading from v2.x

v3.0.0 changes the output contract and, in some cases, the numbers. Read this before
re-running an existing analysis.

**Renamed output.** `threshold` is now `{prefix}_threshold` (`mhw_threshold` /
`mcs_threshold`), so heatwave and cold spell results can be merged. Code doing
`out["threshold"]` must be updated; `output_vars=["threshold"]` still works as an alias.

**New output.** `{prefix}_severity` is returned by default. `detect_mhw_core` and
`mhw_1d_wrapper` now return nine arrays instead of eight, with severity appended last —
code that unpacks the tuple must be updated.

**Results may differ.** Three corrections change detected events and their properties:

- The day-of-year climatology now pools consecutive calendar days around each occurrence,
  as the reference implementation does, instead of days sharing a day-of-year value.
  Thresholds shift by a few thousandths of a degree, which can move event boundaries
  where a day sits marginally above or below the threshold.
- Categories are evaluated at the peak of the severity index rather than at the peak of
  the raw anomaly, and are capped at 4 (Extreme) per Hobday et al. (2018). Previously an
  event could be reported as category 5 or higher.
- Days with an undefined threshold no longer count as exceedances, and rounding noise on
  a zero-variance baseline no longer produces spurious events.

Catalogues produced with v2.x should be regenerated rather than mixed with v3 output.

## Installation

```bash
git clone https://github.com/Gabo2000s/xarray_MHW-xrMHW-.git
cd xarray_MHW-xrMHW-
pip install .
```

Or directly from GitHub:

```bash
pip install git+https://github.com/Gabo2000s/xarray_MHW-xrMHW-.git
```

For development, including the test dependencies:

```bash
pip install -e ".[test]"
```

Requires Python ≥ 3.9. Dependencies (`xarray`, `dask`, `netCDF4`, `scipy`, `pandas`,
`numpy`) are installed automatically; the CI matrix covers Python 3.9, 3.10 and 3.11.

## Quick start

```python
import xarray as xr
import xrMHW

# Detection needs the complete time series for each cell, so chunk over space
# and keep time whole. If time is split into several chunks, xrMHW rechunks it
# to a single chunk and warns; chunk over space to keep memory under control.
ds = xr.open_dataset(
    "GLORYS12V1_thetao_1993-2022.nc",
    chunks={"time": -1, "latitude": 50, "longitude": 50},
)

mhw = xrMHW.xrMHW_func(
    ds,
    temp_var_name="thetao",
    clim_period=(1993, 2022),   # 30-year baseline; required, there is no default
)

mhw.to_netcdf("MHW_output.nc")
```

The same call works unchanged on a 4-D dataset with a `depth` dimension — the extra
dimension is simply another axis to map over.

**Only the continuous fields.** Requesting a subset skips the work the omitted fields
would have required; asking for none of the per-event metrics bypasses the per-event
loop entirely:

```python
sev = xrMHW.xrMHW_func(
    ds, "thetao", (1993, 2022),
    output_vars=["severity", "climatology", "threshold"],
)
```

**Marine cold spells.** The percentile switches to the 10th automatically and the output
variables take the `mcs_` prefix:

```python
mcs = xrMHW.xrMHW_func(ds, "thetao", (1993, 2022), cold_spells=True)
```

**Non-default gap handling.** To bridge short interruptions *before* the duration filter
rather than after it (a documented departure from the standard definition):

```python
joined = xrMHW.xrMHW_func(ds, "thetao", (1993, 2022), join_method="pre-filter")
```

**Climatology and events as two steps.** `xrMHW_clim` computes the seasonal
climatology and threshold on a 366-day `doy` axis; `xrMHW_events` applies a
threshold to a temperature field. Together they give exactly what `xrMHW_func` gives,
and they let you store a climatology, reuse it for several detections, or apply it to
another period or dataset:

```python
clim = xrMHW.xrMHW_clim(ds, "thetao", (1993, 2022))      # dims (..., doy)
clim.to_netcdf("climatology_1993-2022.nc")

mhw = xrMHW.xrMHW_events(ds, "thetao", clim)              # same events as xrMHW_func
new = xrMHW.xrMHW_events(ds_2023, "thetao", clim)         # new data, same baseline
```

The threshold can also be built by hand, per day of year (`doy`, length 366, see
`xrMHW.xrMHW_doy`) or per date (`time`). Which threshold variable it holds,
`mhw_threshold` or `mcs_threshold`, decides whether heatwaves or cold spells are
detected. Running the two steps reads the data twice, so for a single analysis
`xrMHW_func` is faster.

**Baselines.** The baseline sets how the threshold relates to time, and it changes the
results in a warming ocean (Smith et al. 2025, *Prog. Oceanogr.* 231, 103404).
`xrMHW_clim` and `xrMHW_func` take `baseline=`:

| `baseline` | Threshold | Options |
|---|---|---|
| `"fixed"` (default) | one reference period for the whole record | `clim_period` |
| `"detrended"` | follows the long-term trend of the anomaly, fitted over the whole record | `trend="linear"` or `"polynomial"` with `degree` |
| `"adaptation"` | fixed threshold rising at `rate` degC per decade from the centre of the reference period (the climatology stays fixed) | `rate` |
| `"periodic"` | each period of `update_every` years (10) uses the `ref_length` years (30) before it, as recommended by the WMO | `ref_length`, `update_every` |
| `"shifting"` | each year uses its own window: the `window` years (30) before it (`align="lagged"`) or around it (`"centred"`); the trend is removed first by default | `window`, `align`, `detrend_first`, `trend`, `degree` |

```python
detr = xrMHW.xrMHW_func(ds, "thetao", (1993, 2022), baseline="detrended")
adapt = xrMHW.xrMHW_func(ds, "thetao", (1993, 2022), baseline="adaptation", rate=0.1)
```

A wrong trend shape gives spurious events: a linear trend removed from a warming that
accelerates leaves too many events at both ends of the record (Smith et al., Box 1).
`xrMHW.xrMHW_detrend` returns the detrended field itself. With the periodic and
shifting baselines, periods or years without a complete reference window use the
nearest complete one in the record, which makes them a fixed baseline; a warning says
which. Without detrending, a lagged shifting baseline gives more than 10 % of days
above the 90th percentile in a warming record, because each year is compared with a
cooler past (Smith et al., Box 1); `detrend_first=True` (the default) avoids it. The
shifting baseline computes one climatology per distinct window: with 60 years of data
and a 30-year window, its climatology takes about 25 times as long as the fixed one.

### Annual metrics

`xrMHW_annual` turns the daily event fields into one value per cell and calendar year:
the quantities of marineHeatWaves' `blockAverage` with one-year blocks. The result is
small enough to keep in memory even when the detection was not.

```python
ann = xrMHW.xrMHW_annual(mhw)                          # year in place of time
ann["mhw_days"].mean(("latitude", "longitude")).plot()  # event days per year
```

| Variable | Meaning |
|---|---|
| `mhw_count` | events that start in the year |
| `mhw_days` | event days in the calendar year |
| `mhw_duration` | mean duration of the events that start in the year |
| `mhw_intensity_mean`, `mhw_intensity_max`, `mhw_intensity_cumulative` | means over those events of their mean, peak and cumulative intensity |
| `mhw_intensity_max_max` | largest peak among them |
| `mhw_total_icum` | total cumulative intensity of the year |
| `mhw_moderate_days` … `mhw_extreme_days` | event days in each category in the calendar year (needs `mhw_severity`) |

`cum_by` sets where `mhw_total_icum` counts an event that crosses a year:
`"start_year"` (default) puts its whole cumulative intensity in the year it starts, as
`blockAverage` documents; `"calendar_year"` sums the daily intensities within each year.
(`blockAverage`'s code puts such an event in the year it ends; every other quantity
matches it exactly.) Cells without data are NaN when the threshold or the climatology is
in the input; otherwise pass `valid=`, for example `ds["thetao"].notnull().any("time")`.

### Trends

`xrMHW_trend` gives the trend of each cell along a dimension, usually the `year` of
`xrMHW_annual`, with its significance and, optionally, field significance:

```python
tr = xrMHW.xrMHW_trend(ann["mhw_days"], method="sen", test="mk", fdr=0.10)
# slope (per decade), p_value, significant (p < alpha), significant_fdr, n_valid
trs = xrMHW.xrMHW_trend(ann, test="hamed_rao")   # every variable: mhw_days_slope, ...
```

| Option | Values |
|---|---|
| `method` | `"sen"` (Theil-Sen, default) or `"ols"` (least squares) |
| `test` | `"mk"` (Mann-Kendall with tie correction, default), `"hamed_rao"` (corrected for autocorrelation; Hamed & Rao 1998), `"t"` (t-test of the least-squares slope) or `None` |
| `fdr` | `None` or the false discovery rate of Benjamini-Hochberg over the field (Wilks 2016 recommends 0.10) |
| `per` | `"decade"` (default), `"year"` or a number of coordinate units |
| `min_valid` | cells with fewer non-missing values are NaN |

Missing values are skipped, so a mean duration with years without events can be trended.
The results equal `pymannkendall` (`original_test`, `hamed_rao_modification_test`) and
`scipy.stats` (`theilslopes`, `linregress`, `false_discovery_control`) cell by cell.
As in `pymannkendall`, the Hamed-Rao p-value is NaN where the corrected variance is
negative (about 1 % of white-noise series of 33 values). The computation is vectorised
over each block of cells, about 20 times faster than calling `pymannkendall` per cell,
and lazy with Dask, except `fdr`, which ranks the p-values of the whole field and so
computes them.

## Example notebooks

The [`examples/`](examples/) folder has six short notebooks that run on real data
downloaded automatically from Zenodo
([10.5281/zenodo.22909109](https://doi.org/10.5281/zenodo.22909109)):

| Notebook | What it covers |
|---|---|
| [00 · Setup and example data](examples/00_setup_and_data.ipynb) | Installation check, example data, input requirements |
| [01 · A single time series](examples/01_single_point_1D.ipynb) | Detection at one point, the outputs, an event table, validation against `marineHeatWaves` |
| [02 · Sea surface maps](examples/02_sea_surface_3D.ipynb) | Maps of a day, event days per year, marine cold spells |
| [03 · Subsurface marine heatwaves](examples/03_subsurface_4D.ipynb) | Profiles, depth–time diagrams, sections, events hidden below the surface, 3-D view |
| [04 · Many files, Dask and Zarr](examples/04_many_files_and_scaling.ipynb) | `open_mfdataset`, chunk planning from memory, a local cluster, Zarr output |
| [05 · Troubleshooting](examples/05_troubleshooting.ipynb) | Common errors and silent pitfalls, with fixes |

Install their extra dependencies with `pip install ".[examples]"` or with the conda
environment in [`examples/environment.yml`](examples/environment.yml).

## Input requirements

| Dimension | Requirement |
| :--- | :--- |
| **Time** | Daily, datetime-like and convertible with `.toordinal()`. Must be a coordinate/index named `time`. |
| **Space** | Any combination of latitude, longitude and depth; names are not constrained. Depth is optional. |

| Variable | Description | Units |
| :--- | :--- | :--- |
| **Temperature** | Sea surface or potential temperature. | °C (or K — output anomalies inherit the input scale) |

Land and permanently masked cells (all-NaN series) are short-circuited and returned as
NaN, which saves a large fraction of the work on realistic ocean domains. Short data
gaps are linearly interpolated before detection (`max_gap_interp`, 2 days by default);
longer gaps remain NaN and never count as exceedances.

## Outputs

`xrMHW_func` returns an `xarray.Dataset` on the input grid. `{prefix}` is `mhw`, or
`mcs` when `cold_spells=True`. Event fields are NaN-masked outside detected events;
`climatology` and `{prefix}_threshold` are defined at every time step.

| Variable | Description | Units |
| :--- | :--- | :--- |
| `{prefix}_intensity` | Temperature anomaly relative to the seasonal climatology. | °C |
| `{prefix}_severity` | Dimensionless severity index, S = (T − T_clim) / (T_pctile − T_clim). | 1 |
| `{prefix}_duration` | **Local (pointwise)** duration of the exceedance in this cell. | days |
| `{prefix}_category` | 1 = Moderate, 2 = Strong, 3 = Severe, 4 = Extreme (Hobday et al., 2018). | 1 |
| `{prefix}_max_intensity` | Peak anomaly reached during the event. | °C |
| `{prefix}_cum_intensity` | Anomaly integrated over the event. | °C days |
| `climatology` | Day-of-year climatological mean over the baseline, smoothed. | °C |
| `{prefix}_threshold` | Day-of-year climatological percentile over the baseline, smoothed. | °C |

Each variable carries `long_name`, `units` and `description` attributes, and the dataset
records the detection settings in its global attributes (`mhw_detection_period`,
`mhw_percentile`, `mhw_min_duration_days`, `mhw_max_gap_days`, `mhw_join_method`,
`mhw_detection_scope`, `software`) so a saved file documents how it was produced.

The threshold is prefixed because the 90th and 10th percentiles are different fields: a
heatwave run and a cold spell run over the same data can be merged without collision.
`climatology` is shared, since the seasonal mean does not depend on the percentile.

`output_vars` accepts canonical names (`"mhw_severity"`) or short aliases
(`"intensity"`, `"severity"`, `"duration"`, `"category"`, `"max_intensity"`,
`"cum_intensity"`, `"clim"`/`"climatology"`, `"thresh"`/`"threshold"`), as a string or a
sequence. Unknown names raise `ValueError` listing the valid ones.

## Writing results efficiently

Every output has the size of the input grid, and the event fields are NaN outside
events, so a 4-D result written naively is large and mostly empty.

```python
import xrMHW

out = xrMHW.xrMHW_func(ds, "thetao", (1993, 2022), output_vars=["severity", "category"])

# netCDF-4 with compression: event fields are NaN outside events and compress well
enc = {v: {"zlib": True, "complevel": 1} for v in out.data_vars}
out.to_netcdf("mhw_severity.nc", encoding=enc)

# or Zarr (compressed by default; one file per chunk, works with worker processes)
out.to_zarr("mhw_severity.zarr", mode="w")
```

- **Write only the fields you need.** `output_vars` limits what is computed and
  written; each field you leave out saves a full-grid array.
- **Compress.** Written uncompressed, the output can be about 15 times
  larger: for the severity field over the Grand Banks benchmark domain (see
  [`benchmarks/`](benchmarks/)), uncompressed netCDF took 17.2 GiB and Zarr
  1.13 GiB.
- **With `dask.distributed` worker processes, prefer Zarr.** Writing a large netCDF file
  from worker processes reopens the file for every chunk under a lock, and in our
  benchmarks it failed after about 5 GB; Zarr writes each chunk independently.
- **Keep the time axis in one chunk.** The climatology and the event detection use each
  cell's complete record. A time axis split into several chunks is rechunked to one
  chunk, with a warning; chunk over the spatial dimensions to control memory.

## Parameters

All of the following are passed as keyword arguments to `xrMHW_func`.

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `clim_period` | *(required)* | `(start_year, end_year)` of the climatology baseline, inclusive. |
| `output_vars` | `None` (all) | Which output variables to compute and return. |
| `cold_spells` | `False` | Detect marine cold spells instead of heatwaves. |
| `pctile` | `90` / `10` | Percentile threshold; defaults to 10 when `cold_spells=True`. |
| `min_duration` | `5` | Minimum event length, in days. |
| `window_half_width` | `5` | Half-width of the climatology window (total 2w + 1 = 11 days). |
| `smooth_pctile` | `True` | Smooth climatology and threshold with a moving average. |
| `smooth_width` | `31` | Width of that smoothing window, in days. |
| `max_gap_interp` | `2` | Longest run of missing values filled by interpolation before detection. |
| `join_across_gaps` | `True` | Join events separated by a brief return to normal conditions. |
| `max_gap` | `2` | Longest such gap that is bridged, in days. |
| `join_method` | `'post-filter'` | Order of joining relative to the duration filter (see below). |

## Methodology

### Climatology and threshold

For each day of the year, the climatological mean and the percentile threshold are
computed by pooling the ±`window_half_width` **consecutive calendar days** around every
occurrence of that day within the baseline period, across all baseline years, and are
then smoothed with a `smooth_width`-day moving average. 29 February is interpolated from
its neighbours rather than estimated from the roughly four times fewer samples available
for it, and all years are aligned to a 366-day grid so that a given calendar date always
maps to the same index.

The distinction between pooling *consecutive calendar days* and pooling *days sharing a
day-of-year value* is not cosmetic: the latter excludes 29 February from neighbouring
windows and shifts the threshold by a few thousandths of a degree. xrMHW uses the
former, which is what the reference implementation does.

### Event definition and gap handling

A day is an exceedance when the temperature is above the threshold (below it, for cold
spells). Runs of exceedances are labelled, and runs shorter than `min_duration` are
discarded.

By default (`join_method='post-filter'`) **the minimum-duration filter is applied first,
and surviving events separated by `max_gap` days or fewer are then joined** — the order
used by Hobday et al. (2016) and by the reference implementation. Setting
`join_method='pre-filter'` bridges short gaps *before* the duration filter, so a long
event fragmented by a brief dip survives as a single event instead of being discarded as
two short ones. This is a deliberate, documented departure from the standard definition,
and it is not the default: use it when the fragmentation is a known artefact of the
data, and say so in your methods.

### Severity index and categories

`{prefix}_severity` is the dimensionless severity index

> S = (T − T_clim) / (T_p90 − T_clim)

following Sen Gupta et al. (2020) and Malan et al. (2025). Normalising the anomaly by
the local threshold-to-mean distance makes it comparable across depths and regions in a
way a raw anomaly in °C is not: where variability is small, a modest anomaly can
represent far more extreme conditions than a larger anomaly in a variable surface layer.
The same expression holds for cold spells, where numerator and denominator are both
negative.

It is the continuous form of the Hobday et al. (2018) categorisation: the category is
`floor(S)` evaluated at the peak of the severity index — not at the peak of the raw
anomaly, which is a different day whenever the threshold-to-mean distance varies within
the event — and is capped at 4 (Extreme).

## Validation against the reference implementation

In its default configuration xrMHW reproduces
[`marineHeatWaves`](https://github.com/ecjoliver/marineHeatWaves) to floating-point
precision. On a seeded 30-year daily series with injected events of known length:

| Quantity | Agreement with the reference |
| :--- | :--- |
| Percentile threshold | max abs difference **0.0 °C** (exact) |
| Seasonal climatology | max abs difference ~7 × 10⁻¹⁵ °C (machine precision) |
| Event count, start and end dates | identical |
| Durations | identical |
| Peak and cumulative intensity | identical to < 10⁻⁸ °C |
| Categories | identical |

These are assertions in `tests/test_against_marineHeatWaves.py`, checked on every push
by CI, with a climatology tolerance of 10⁻¹⁰ °C. A separate test pins the optimised,
loop-free climatology against a literal transcription of the definition, so the
implementation cannot be made faster at the cost of silently drifting from it.

## Performance

Detection runs as a Python function mapped over the spatial dimensions by
`xarray.apply_ufunc` and executed in parallel across Dask chunks, so it scales with the
available cores and streams datasets larger than memory. Two costs that dominate a naive
implementation are avoided: the day-of-year climatology is computed without a
366-iteration loop, using a single percentile call over a strided gather, and the
calendar arithmetic — identical for every cell — is computed once per time axis instead
of once per cell.

On a 30-year daily series this is of the order of **7 ms per grid cell on a single
core**, roughly an order of magnitude faster than the previous release while returning
bit-identical results. Absolute timings depend on hardware, record length and chunking;
measure on your own domain before planning a large run.

## Related software

| Package | Language | Grids | Notes |
| :--- | :--- | :--- | :--- |
| [`marineHeatWaves`](https://github.com/ecjoliver/marineHeatWaves) | Python | single time series | The reference implementation of Hobday et al. (2016); xrMHW's detection logic is adapted from it and validated against it. |
| [`heatwaveR`](https://github.com/robwschlegel/heatwaveR) | R / C++ | single time series | Schlegel & Smit (2018); the established R implementation, with extensive event statistics and plotting. |
| [`xmhw`](https://github.com/coecms/xmhw) | Python | xarray, gridded | Also derived from Oliver's code; oriented to sea surface temperature. |
| **xrMHW** | Python | xarray + Dask, N-D | Depth-resolved 3-D/4-D grids, out-of-core execution, severity index and cold spells as first-class outputs, selective outputs, bit-exact reproduction of the reference. |

If your data are a single time series or a surface field that fits in memory, the
packages above are mature and well documented, and there is no reason to prefer xrMHW.
Its case is the depth-resolved, out-of-core one.

## Testing

```bash
pip install -e ".[test]"
pytest -v
```

The suite covers agreement with the reference implementation, the xarray/Dask interface,
gap handling in both modes, land masks, cold spells, the severity index and category
definitions, output selection, Dask/eager equivalence, and the degenerate zero-variance
case. `tests/marineHeatWaves.py` is a vendored copy of the reference implementation used
only for comparison.

## Contributing

Bug reports, numerical discrepancies and feature requests are welcome through the
[issue tracker](https://github.com/Gabo2000s/xarray_MHW-xrMHW-/issues). See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, coding conventions and
test requirements.

## How to cite

If xrMHW contributes to work you publish, please cite the archived release:

> Gutiérrez-Cárdenas, G. S. (2026). *xrMHW: A High-Performance Multidimensional Marine
> Heatwave Detector* (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.18476144

and the definition it implements (Hobday et al., 2016). Please also consider citing
Oliver's `marineHeatWaves`, from which the detection logic is adapted.

## License

Distributed under the GPL-3.0 license; see [LICENSE](LICENSE). `tests/marineHeatWaves.py`
is redistributed from Eric C. J. Oliver's `marineHeatWaves` under the same license, with
a one-line compatibility patch for NumPy ≥ 1.25, and is used only to validate xrMHW
against it.

## References

Hobday, A. J., Alexander, L. V., Perkins, S. E., Smale, D. A., Straub, S. C.,
Oliver, E. C. J., et al. (2016). A hierarchical approach to defining marine heatwaves.
*Progress in Oceanography*, 141, 227–238. https://doi.org/10.1016/j.pocean.2015.12.014

Hobday, A. J., Oliver, E. C. J., Sen Gupta, A., Benthuysen, J. A., Burrows, M. T.,
Donat, M. G., et al. (2018). Categorizing and naming marine heatwaves. *Oceanography*,
31(2), 162–173. https://doi.org/10.5670/oceanog.2018.205

Malan, N., Sen Gupta, A., Schaeffer, A., Zhang, S., Doblin, M. A., Pilo, G. S., et al.
(2025). Lifting the lid on marine heatwaves. *Progress in Oceanography*, 239, 103539.
https://doi.org/10.1016/j.pocean.2025.103539

Schlegel, R. W., & Smit, A. J. (2018). heatwaveR: A central algorithm for the detection
of heatwaves and cold-spells. *Journal of Open Source Software*, 3(27), 821.
https://doi.org/10.21105/joss.00821

Sen Gupta, A., Thomsen, M., Benthuysen, J. A., Hobday, A. J., Oliver, E. C. J.,
Alexander, L. V., et al. (2020). Drivers and impacts of the most extreme marine heatwave
events. *Scientific Reports*, 10, 19359. https://doi.org/10.1038/s41598-020-75445-3

Sun, D., Li, F., Jing, Z., Hu, S., & Zhang, B. (2023). Frequent marine heatwaves hidden
below the surface of the global ocean. *Nature Geoscience*, 16, 1099–1104.
https://doi.org/10.1038/s41561-023-01325-w
