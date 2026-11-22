"""
xrMHW — N-dimensional Marine Heatwave detection with xarray + Dask
==================================================================

An efficient, parallel, out-of-core implementation of the pointwise marine
heatwave (MHW) definition of Hobday et al. (2016), operating natively on
N-dimensional grids (time x latitude x longitude x depth) via
``xarray.apply_ufunc`` and Dask.

Scope
-----
xrMHW computes *pointwise* fields: every grid cell is treated as an
independent 1-D time series, exactly as in the reference implementation
``marineHeatWaves`` (Oliver, https://github.com/ecjoliver/marineHeatWaves).
It does **not** perform spatial/volumetric object tracking; event-level
(object) metrics of coherent 3-D structures are outside its scope.

Reproducibility
---------------
In its default configuration (``join_method='post-filter'``) the seasonal
climatology, the percentile threshold, the detected events and their
categories reproduce ``marineHeatWaves`` to floating-point precision.

References
----------
Hobday, A. J., et al. (2016). A hierarchical approach to defining marine
    heatwaves. Progress in Oceanography, 141, 227-238.
    https://doi.org/10.1016/j.pocean.2015.12.014
Hobday, A. J., et al. (2018). Categorizing and naming marine heatwaves.
    Oceanography, 31(2), 162-173. https://doi.org/10.5670/oceanog.2018.205
Sen Gupta, A., et al. (2020). Drivers and impacts of the most extreme marine
    heatwave events. Scientific Reports, 10, 19359.
    https://doi.org/10.1038/s41598-020-75445-3
Malan, N., et al. (2025). Lifting the lid on marine heatwaves.
    Progress in Oceanography, 239, 103539.
    https://doi.org/10.1016/j.pocean.2025.103539

Author: Gutiérrez-Cárdenas, G. S. (ORCID: 0000-0002-3915-7684)
License: GPL-3.0
"""

import warnings
from datetime import date

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import xarray as xr

__version__ = "3.0.0"

__all__ = ["detect_mhw_core", "mhw_1d_wrapper", "xrMHW_func"]

# Number of days on the 366-day (leap-year) day-of-year grid.
_NDOY = 366
# Day-of-year index of 29 February on that grid (1-based).
_FEB29 = 60


# =============================================================================
# 0. INTERNAL HELPERS
# =============================================================================

def _leap_doy_map():
    """Month/day -> day-of-year on a 366-day grid, built once per process.

    Aligning every year (leap or not) to the leap-year grid is what makes
    1 March always index 61, matching the reference implementation.
    """
    if _leap_doy_map._cache is None:
        t_leap = np.arange(date(2012, 1, 1).toordinal(),
                           date(2012, 12, 31).toordinal() + 1)
        dates_leap = pd.to_datetime([date.fromordinal(int(x)) for x in t_leap])
        _leap_doy_map._cache = {
            (m, d): dy for m, d, dy in
            zip(dates_leap.month, dates_leap.day, range(1, _NDOY + 1))
        }
    return _leap_doy_map._cache


_leap_doy_map._cache = None


def _time_metadata(t_ordinal):
    """Return ``(year, doy)`` for an ordinal time vector.

    ``apply_ufunc`` passes the *same* time vector for every grid cell, so the
    calendar arithmetic is memoised: it is executed once per distinct time
    axis instead of once per pixel. The cache key is cheap to compute and
    collisions are impossible for a fixed-length contiguous daily axis.
    """
    key = (int(t_ordinal[0]), int(t_ordinal[-1]), len(t_ordinal))
    cached = _time_metadata._cache.get(key)
    if cached is not None:
        return cached

    dates_pd = pd.to_datetime([date.fromordinal(int(x)) for x in t_ordinal])
    year = dates_pd.year.values
    dmap = _leap_doy_map()
    doy = np.array([dmap.get((m, d), 0)
                    for m, d in zip(dates_pd.month.values, dates_pd.day.values)])

    # Keep the cache tiny: a worker only ever sees a handful of time axes.
    if len(_time_metadata._cache) > 8:
        _time_metadata._cache.clear()
    _time_metadata._cache[key] = (year, doy)
    return year, doy


_time_metadata._cache = {}


def _runavg(ts, w):
    """Centred moving average with periodic (circular) extension."""
    n = len(ts)
    return np.convolve(np.concatenate((ts, ts, ts)), np.ones(w) / w,
                       mode='same')[n:2 * n]


def _seasonal_climatology(temp, doy, clim_mask, pctile, window_half_width,
                          smooth_pctile, smooth_width):
    """Day-of-year seasonal mean and percentile threshold.

    Windowing follows Hobday et al. (2016) as implemented by Oliver: for each
    occurrence of a day-of-year within the baseline, the +/- ``window_half_width``
    CONSECUTIVE calendar days around it are pooled (an index-space window),
    across all baseline years. A day-of-year *value* window would exclude
    29 February from neighbouring windows and shift the threshold by a few
    thousandths of a degree.

    This is a vectorised reformulation of that definition: the per-day-of-year
    loop is replaced by a single strided gather plus one percentile call.
    The percentile does not depend on the order of the samples, so the threshold
    is identical to the loop bit for bit. The mean does, in its last bits: a
    floating-point sum depends on the order of its terms, so the climatology
    agrees with the loop to rounding (~1e-15 degC). Both are checked in
    ``tests/test_against_marineHeatWaves.py``.

    Returns
    -------
    (seas_clim_year, thresh_clim_year) : tuple of np.ndarray
        Two arrays of length 366, already interpolated at 29 February and
        smoothed if requested.
    """
    T = len(temp)
    w = window_half_width
    clim_idx = np.flatnonzero(clim_mask)
    cs, ce = clim_idx[0], clim_idx[-1]

    # Rows of `win` are the index-space windows centred on each time step;
    # the NaN padding reproduces the loop's truncation at the record edges.
    padded = np.concatenate([np.full(w, np.nan), temp, np.full(w, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1)

    rows = np.arange(cs, ce + 1)
    doy_rows = doy[cs:ce + 1]

    # Position of each occurrence within its day-of-year group, so the windows
    # can be scattered into a dense (doy, occurrence, offset) cube.
    order = np.argsort(doy_rows, kind='stable')
    sorted_doy = doy_rows[order]
    rank_sorted = (np.arange(len(sorted_doy))
                   - np.searchsorted(sorted_doy, sorted_doy, side='left'))
    ranks = np.empty_like(rank_sorted)
    ranks[order] = rank_sorted

    counts = np.bincount(doy_rows, minlength=_NDOY + 1)
    pool = np.full((_NDOY + 1, counts.max(), 2 * w + 1), np.nan)
    pool[doy_rows, ranks, :] = win[rows, :]
    pool = pool.reshape(_NDOY + 1, -1)

    thresh = np.full(_NDOY + 1, np.nan)
    seas = np.full(_NDOY + 1, np.nan)

    clean = ~np.isnan(pool).any(axis=1) & (counts > 0)
    dirty = (~clean) & (counts > 0)
    if clean.any():
        thresh[clean] = np.percentile(pool[clean], pctile, axis=1)
        seas[clean] = pool[clean].mean(axis=1)
    if dirty.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            thresh[dirty] = np.nanpercentile(pool[dirty], pctile, axis=1)
            seas[dirty] = np.nanmean(pool[dirty], axis=1)

    thresh_clim_year = thresh[1:_NDOY + 1]
    seas_clim_year = seas[1:_NDOY + 1]

    # 29 February is interpolated from its neighbours
    thresh_clim_year[_FEB29 - 1] = 0.5 * (thresh_clim_year[_FEB29 - 2]
                                          + thresh_clim_year[_FEB29])
    seas_clim_year[_FEB29 - 1] = 0.5 * (seas_clim_year[_FEB29 - 2]
                                        + seas_clim_year[_FEB29])

    if smooth_pctile:
        thresh_clim_year = _runavg(thresh_clim_year, smooth_width)
        seas_clim_year = _runavg(seas_clim_year, smooth_width)

    return seas_clim_year, thresh_clim_year


def _filter_min_duration(events, n_events, min_duration):
    """Set to 0 every labeled event shorter than `min_duration` (vectorized)."""
    if n_events == 0:
        return events
    counts = np.bincount(events.ravel())
    too_short = np.nonzero(counts < min_duration)[0]
    too_short = too_short[too_short != 0]  # never drop the background label
    if too_short.size:
        events = np.where(np.isin(events, too_short), 0, events)
    return events


def _bridge_gaps(mask, max_gap):
    """Fill interior False runs of length <= `max_gap` so the flanking True
    runs become a single continuous event. Leading/trailing gaps are never
    filled (there is nothing to join them to)."""
    mask = mask.copy()
    gaps, n_gaps = ndimage.label(~mask)
    if n_gaps:
        for sl in ndimage.find_objects(gaps):
            start, stop = sl[0].start, sl[0].stop
            if (stop - start) <= max_gap and start > 0 and stop < len(mask):
                mask[start:stop] = True
    return mask


# =============================================================================
# 1. CORE SCIENTIFIC LOGIC (NumPy/pandas based)
# =============================================================================

def detect_mhw_core(t_ordinal, temp, clim_period=(None, None), pctile=90,
                    window_half_width=5, smooth_pctile=True, smooth_width=31,
                    min_duration=5, join_across_gaps=True, max_gap=2,
                    cold_spells=False, compute_event_metrics=True,
                    join_method='post-filter'):
    """
    Core detection algorithm for marine heatwaves (or cold spells) on a single
    1-D time series, following Hobday et al. (2016). Adapted from
    https://github.com/ecjoliver/marineHeatWaves (Oliver).

    In its default configuration this reproduces the reference implementation
    to floating-point precision. Gap handling is selectable via `join_method`:

      - 'post-filter' (DEFAULT, = Hobday/Oliver): apply the minimum-duration
        filter first, then join surviving events separated by <= `max_gap` days.
      - 'pre-filter' (xrMHW option): join short gaps BEFORE the duration filter,
        so a fragmented long event survives as one. This is a deliberate,
        documented departure from Hobday/Oliver, not the default.

    Parameters
    ----------
    t_ordinal : np.ndarray
        Time vector in ordinal format (integers), daily resolution.
    temp : np.ndarray
        Temperature vector (1-D).
    clim_period : tuple
        (start_year, end_year) for the climatology baseline.
    pctile : int or float
        Percentile threshold (90 for MHW, typically 10 for MCS).
    window_half_width : int
        Half-width of the climatology window in days (default 5, i.e. 11 days).
    smooth_pctile : bool
        Whether to smooth the climatology and threshold (default True).
    smooth_width : int
        Width of the moving-average smoothing window in days (default 31).
    min_duration : int
        Minimum duration for acceptance of a detected event (default 5 days).
    join_across_gaps : bool
        Whether to join events separated by <= `max_gap` days (default True).
    max_gap : int
        Maximum gap length in days allowed to join two events (default 2).
    cold_spells : bool
        If True, detect marine cold spells (temp < threshold). Default False.
    compute_event_metrics : bool
        If False, skip the per-event metric loop and return only the event
        mask, the climatology, the threshold, the anomaly and the severity.
        The four per-event arrays are returned as zeros. Use when only the
        continuous fields are needed; it removes the only Python-level loop
        that scales with the number of events.
    join_method : {'post-filter', 'pre-filter'}
        Order of gap-joining relative to the duration filter (default
        'post-filter', which matches Hobday/Oliver). See summary above.

    Returns
    -------
    tuple of np.ndarray
        (clim_seas, clim_thresh, anomaly, is_mhw, mhw_duration, mhw_category,
         mhw_intensity_max, mhw_intensity_cum, mhw_severity)

    Notes
    -----
    `mhw_duration` is the *local* persistence of the exceedance in this time
    series. On a 4-D grid each (lat, lon, depth) cell is independent, so
    neighbouring cells may report different durations for what is physically
    one structure. It is not the lifetime of a tracked 3-D object.
    """
    T = len(t_ordinal)
    year, doy = _time_metadata(t_ordinal)

    # --- Climatology -------------------------------------------------------
    if (clim_period[0] is None) or (clim_period[1] is None):
        clim_start, clim_end = year[0], year[-1]
    else:
        clim_start, clim_end = clim_period

    clim_mask = (year >= clim_start) & (year <= clim_end)
    if not clim_mask.any():
        raise ValueError(
            f"No data in the climatology baseline {clim_start}-{clim_end}; "
            f"the record spans {year[0]}-{year[-1]}."
        )

    seas_clim_year, thresh_clim_year = _seasonal_climatology(
        temp, doy, clim_mask, pctile, window_half_width,
        smooth_pctile, smooth_width)

    clim_thresh = thresh_clim_year[doy - 1]
    clim_seas = seas_clim_year[doy - 1]

    # --- Continuous diagnostics -------------------------------------------
    anomaly = temp - clim_seas

    # Severity index S = (T - T_clim) / (T_pctile - T_clim), dimensionless
    # (Sen Gupta et al. 2020; Malan et al. 2025, eq. 1). It is the continuous
    # form of the Hobday et al. (2018) categories: floor(S) gives 1 = Moderate,
    # 2 = Strong, 3 = Severe, >=4 = Extreme. Because it is normalised by the
    # local threshold-to-mean distance it is comparable across depths and
    # regions, which a raw anomaly in degC is not. The same expression is
    # valid for cold spells: numerator and denominator are both negative.
    denom = clim_thresh - clim_seas
    with np.errstate(divide='ignore', invalid='ignore'):
        severity = np.where(np.abs(denom) > 1e-10, anomaly / denom, np.nan)

    # --- Event detection ---------------------------------------------------
    excess = (clim_thresh - temp) if cold_spells else (temp - clim_thresh)

    # The comparison is `excess > 0` (Hobday/Oliver), guarded by a few units in
    # the last place. A degenerate baseline with no variability makes the
    # smoothed mean and percentile coincide to within ~1e-15 degC, and an
    # unguarded `>` would then flag the whole record as one event on pure
    # rounding noise. The tolerance scales with the magnitude of the threshold
    # and stays ~1e-14 degC for ocean temperatures: many orders of magnitude
    # below the precision of any observational or model dataset, so genuine
    # low-variance (e.g. abyssal, milli-kelvin) exceedances are untouched.
    tol = 8 * np.finfo(float).eps * np.maximum(np.abs(clim_thresh), 1.0)
    # NaN climatology days (e.g. absent DOYs) must never count as exceedances.
    exceed_bool = np.asarray(excess > tol) & ~np.isnan(clim_thresh)

    if join_across_gaps and join_method == 'pre-filter':
        # xrMHW option: bridge short gaps BEFORE the duration filter, so an
        # otherwise-fragmented long event survives as one. Deliberate,
        # documented departure from Hobday/Oliver.
        exceed_bool = _bridge_gaps(exceed_bool, max_gap)
        events, n_events = ndimage.label(exceed_bool)
        events = _filter_min_duration(events, n_events, min_duration)
    else:
        # Hobday/Oliver default ('post-filter'): duration filter FIRST, then
        # join surviving events across gaps <= max_gap. Joined events are
        # >= min_duration by construction, so no second filter is needed.
        events, n_events = ndimage.label(exceed_bool)
        events = _filter_min_duration(events, n_events, min_duration)
        if join_across_gaps:
            survivors = _bridge_gaps(events > 0, max_gap)
            events, n_events = ndimage.label(survivors)

    # --- Per-event metrics -------------------------------------------------
    mhw_duration = np.zeros(T, dtype=float)
    mhw_category = np.zeros(T, dtype=float)
    mhw_intensity_max = np.zeros(T, dtype=float)
    mhw_intensity_cum = np.zeros(T, dtype=float)

    if not compute_event_metrics:
        # Fast path: the duration filter already removed short events, so the
        # surviving labels are exactly the event mask.
        is_mhw = events != 0
        return (clim_seas, clim_thresh, anomaly, is_mhw, mhw_duration,
                mhw_category, mhw_intensity_max, mhw_intensity_cum, severity)

    is_mhw = np.zeros(T, dtype=bool)
    active_labels = np.unique(events)
    active_labels = active_labels[active_labels != 0]

    for ev_id in active_labels:
        idx = np.where(events == ev_id)[0]
        dur = len(idx)
        if dur < min_duration:  # defensive; the filter above guarantees this
            continue

        anoms = anomaly[idx]
        sev = severity[idx]

        # Intensity peak: extreme of the raw anomaly (Oliver: tt_peak).
        peak_idx = np.argmin(anoms) if cold_spells else np.argmax(anoms)
        i_max = anoms[peak_idx]
        i_cum = np.sum(anoms)

        # Category peak: extreme of the NORMALISED excess (Oliver: tt_peakCat).
        # This is a different time step from the intensity peak whenever the
        # threshold-to-mean distance varies within the event, which it does
        # seasonally. Capped at 4 (Extreme) per Hobday et al. (2018).
        if np.all(np.isnan(sev)):
            cat = 1.0
        else:
            cat_peak = np.nanargmax(sev)
            cat = min(max(1, int(np.floor(sev[cat_peak]))), 4)

        mhw_duration[idx] = float(dur)
        mhw_category[idx] = float(cat)
        mhw_intensity_max[idx] = i_max
        mhw_intensity_cum[idx] = i_cum
        is_mhw[idx] = True

    return (clim_seas, clim_thresh, anomaly, is_mhw, mhw_duration,
            mhw_category, mhw_intensity_max, mhw_intensity_cum, severity)


# =============================================================================
# 2. ROBUST WRAPPER (NaN handling & vectorisation interface)
# =============================================================================

def mhw_1d_wrapper(time_ordinal, temp, clim_start_year, clim_end_year,
                   max_gap_interp=2, **kwargs):
    """
    Wrapper handling NaNs and the ``xarray.apply_ufunc`` interface, passing
    configuration through to the core detection logic.

    Parameters
    ----------
    time_ordinal : np.ndarray
        Time vector in ordinal format (integers).
    temp : np.ndarray
        Temperature vector (1-D).
    clim_start_year, clim_end_year : int
        Bounds of the climatology baseline.
    max_gap_interp : int, optional
        Maximum run of consecutive missing values (NaNs) to fill by linear
        interpolation before detection. Longer gaps remain NaN.
    **kwargs
        Passed to `detect_mhw_core`.

    Returns
    -------
    tuple of np.ndarray
        The nine arrays returned by `detect_mhw_core`.
    """
    T = len(time_ordinal)
    temp = temp.copy()  # Dask hands over read-only buffers

    # 1. Fast exit for land / fully masked cells.
    if np.isnan(temp).all():
        nan_arr = np.full(T, np.nan)
        return (nan_arr, nan_arr, nan_arr, np.zeros(T, dtype=bool),
                nan_arr, nan_arr, nan_arr, nan_arr, nan_arr)

    # 2. Interpolate only short data gaps.
    if np.isnan(temp).any():
        valid_indices = np.flatnonzero(~np.isnan(temp))
        if len(valid_indices) > 1:
            gaps = np.diff(valid_indices)
            fillable = np.where((gaps > 1) & (gaps <= (max_gap_interp + 1)))[0]
            for i in fillable:
                start_idx, end_idx = valid_indices[i], valid_indices[i + 1]
                x_gap = np.arange(start_idx + 1, end_idx)
                temp[x_gap] = np.interp(x_gap, [start_idx, end_idx],
                                        [temp[start_idx], temp[end_idx]])

    return detect_mhw_core(
        time_ordinal.astype(int),
        temp.astype(float),
        clim_period=(int(clim_start_year), int(clim_end_year)),
        **kwargs
    )


# =============================================================================
# 3. XARRAY INTEGRATION
# =============================================================================

# name -> (long_name, units, description) for the output dataset.
_VAR_ATTRS = {
    'intensity': (
        'MHW intensity (temperature anomaly)', 'degC',
        'Temperature anomaly relative to the seasonal climatology, on days '
        'belonging to a detected event.'),
    'severity': (
        'MHW severity index', '1',
        'Dimensionless severity S = (T - T_clim) / (T_pctile - T_clim) '
        '(Sen Gupta et al. 2020; Malan et al. 2025). Comparable across depths '
        'and regions; floor(S) gives the Hobday et al. (2018) category.'),
    'duration': (
        'Local (pointwise) duration of the exceedance', 'days',
        'Length of the event detected in THIS grid cell, treated as an '
        'independent 1-D time series, broadcast over the days of the event. '
        'On a 4-D grid neighbouring cells may differ; this is not the '
        'lifetime of a tracked 3-D object.'),
    'category': (
        'MHW category (Hobday et al. 2018)', '1',
        '1 = Moderate, 2 = Strong, 3 = Severe, 4 = Extreme. Evaluated at the '
        'peak of the severity index and capped at 4.'),
    'max_intensity': (
        'Peak intensity of the event', 'degC',
        'Largest temperature anomaly reached during the event.'),
    'cum_intensity': (
        'Cumulative intensity of the event', 'degC days',
        'Temperature anomaly integrated over the duration of the event.'),
    'climatology': (
        'Seasonal climatological mean temperature', 'degC',
        'Day-of-year climatological mean over the baseline period, smoothed.'),
    'threshold': (
        'Seasonal percentile threshold', 'degC',
        'Day-of-year climatological percentile over the baseline period, '
        'smoothed. 90th percentile for MHW, 10th for MCS.'),
}

# Fields that require the per-event metric loop.
_EVENT_METRIC_FIELDS = ('duration', 'category', 'max_intensity', 'cum_intensity')


def _canonical_outputs(prefix):
    """Canonical output variable names, in dataset order."""
    return [f'{prefix}_intensity', f'{prefix}_severity', f'{prefix}_duration',
            f'{prefix}_category', f'{prefix}_max_intensity',
            f'{prefix}_cum_intensity', 'climatology', f'{prefix}_threshold']


def _resolve_output_vars(output_vars, prefix):
    """Resolve requested output names to canonical dataset variable names.

    Accepts canonical names (``'mhw_severity'``), short aliases
    (``'intensity'``, ``'severity'``, ``'duration'``, ``'category'``,
    ``'max_intensity'``, ``'cum_intensity'``, ``'clim'``/``'climatology'``,
    ``'thresh'``/``'threshold'``), a single string, or a sequence.
    ``output_vars=None`` returns every variable.
    """
    canonical = _canonical_outputs(prefix)
    if output_vars is None:
        return list(canonical)

    aliases = {
        'intensity': f'{prefix}_intensity',
        'severity': f'{prefix}_severity',
        'duration': f'{prefix}_duration',
        'category': f'{prefix}_category',
        'max_intensity': f'{prefix}_max_intensity',
        'cum_intensity': f'{prefix}_cum_intensity',
        'climatology': 'climatology',
        'clim': 'climatology',
        # 'threshold' stays accepted for backward compatibility even though the
        # variable is now prefixed, so MHW and MCS outputs can be merged.
        'threshold': f'{prefix}_threshold',
        'thresh': f'{prefix}_threshold',
    }
    if isinstance(output_vars, str):
        output_vars = [output_vars]

    resolved = []
    for v in output_vars:
        key = v.strip()
        if key in canonical:
            name = key
        elif key in aliases:
            name = aliases[key]
        elif f'{prefix}_{key}' in canonical:
            name = f'{prefix}_{key}'
        else:
            raise ValueError(
                f"Unknown output var '{v}'. Choose from {canonical} "
                f"or aliases {sorted(aliases)}.")
        if name not in resolved:
            resolved.append(name)
    return resolved


def xrMHW_func(ds, temp_var_name, clim_period, output_vars=None, **kwargs):
    """
    Detect marine heatwaves (or cold spells) over an entire xarray Dataset,
    parallelised with Dask.

    Each grid cell is processed as an independent 1-D time series following
    Hobday et al. (2016); the detection is mapped over the spatial dimensions
    with ``xarray.apply_ufunc``, so the computation streams over Dask chunks
    and scales to datasets larger than memory.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a temperature variable and a 'time' coordinate.
        Chunk so that time is contiguous, e.g. ``chunks={'time': -1}``.
        If the time axis is split into several chunks, it is rechunked to a
        single chunk with a warning.
    temp_var_name : str
        Name of the temperature variable in `ds`.
    clim_period : tuple of (int, int)
        Start and end years (inclusive) of the climatology baseline, e.g.
        ``(1991, 2020)`` for the current WMO normal.
    output_vars : str or sequence of str, optional
        Which variables to return. ``None`` (default) returns all of them.
        Restricting the selection also skips the work needed for the omitted
        fields: asking only for continuous fields (e.g.
        ``['severity', 'climatology']``) bypasses the per-event metric loop.
    **kwargs
        Passed to the detection logic:

        * **cold_spells** (bool, default False) — detect marine cold spells.
        * **pctile** (int/float, default 90 for MHW, 10 for MCS).
        * **min_duration** (int, default 5) — minimum event length in days.
        * **window_half_width** (int, default 5) — climatology window is
          ``2*w + 1`` days.
        * **smooth_pctile** (bool, default True) and **smooth_width**
          (int, default 31) — smoothing of climatology and threshold.
        * **max_gap_interp** (int, default 2) — longest run of missing values
          filled by interpolation *before* detection.
        * **join_across_gaps** (bool, default True) and **max_gap**
          (int, default 2) — joining of events separated by a brief return to
          normal conditions.
        * **join_method** ({'post-filter', 'pre-filter'}, default
          'post-filter') — order of joining relative to the duration filter.

    Returns
    -------
    xr.Dataset
        Dataset on the input grid. Event fields are NaN where no event is
        detected; `climatology` and `{prefix}_threshold` are defined at every
        time step. `{prefix}` is 'mhw', or 'mcs' when ``cold_spells=True``:

        * ``{prefix}_intensity`` — anomaly relative to climatology (degC)
        * ``{prefix}_severity`` — dimensionless severity index
        * ``{prefix}_duration`` — local pointwise duration (days)
        * ``{prefix}_category`` — 1 Moderate ... 4 Extreme
        * ``{prefix}_max_intensity`` — peak anomaly of the event (degC)
        * ``{prefix}_cum_intensity`` — integrated anomaly (degC days)
        * ``climatology`` — seasonal climatological mean (degC)
        * ``{prefix}_threshold`` — seasonal percentile threshold (degC)

    Notes
    -----
    Outputs are pointwise. `{prefix}_duration` is the persistence of the
    exceedance in each individual cell, not the lifetime of a coherent 3-D
    structure; object-level tracking is out of scope for this package.
    """
    if 'time' not in ds.indexes:
        raise ValueError("Input dataset must have a 'time' coordinate/index.")
    if temp_var_name not in ds:
        raise KeyError(f"Variable '{temp_var_name}' not found in the dataset. "
                       f"Available: {list(ds.data_vars)}")

    # The climatology and the event detection use each cell's complete record,
    # so apply_ufunc needs the whole time axis in one chunk; without this it
    # raises a ValueError that does not say what to do.
    da = ds[temp_var_name]
    if da.chunks is not None and 'time' in da.dims:
        t_axis = da.get_axis_num('time')
        if len(da.chunks[t_axis]) > 1:
            other = [max(c) for i, c in enumerate(da.chunks) if i != t_axis]
            new_mib = (da.sizes['time'] * int(np.prod(other))
                       * np.dtype(da.dtype).itemsize / 2 ** 20)
            warnings.warn(
                "xrMHW needs the whole time axis in one chunk (the climatology "
                "and the event detection use each cell's complete record); "
                f"rechunking with chunks={{'time': -1}} (~{new_mib:.0f} MiB per "
                "chunk). To control memory, chunk the input yourself, e.g. "
                "ds.chunk({'time': -1, 'latitude': 10, 'longitude': 10}).",
                UserWarning, stacklevel=2)
            ds = ds.chunk({'time': -1})

    time_index = ds.indexes['time']
    try:
        time_ordinal_np = time_index.map(lambda x: x.toordinal()).values
    except AttributeError:
        raise TypeError("Time coordinate must be convertible to ordinal "
                        "(datetime-like).")

    time_ordinal_da = xr.DataArray(time_ordinal_np,
                                   coords={'time': ds['time']}, dims='time')

    func_kwargs = {'clim_start_year': clim_period[0],
                   'clim_end_year': clim_period[1]}
    func_kwargs.update(kwargs)
    func_kwargs.setdefault('cold_spells', False)
    # 90th percentile for heatwaves, 10th for cold spells.
    func_kwargs.setdefault('pctile', 10 if func_kwargs['cold_spells'] else 90)

    prefix = 'mcs' if func_kwargs['cold_spells'] else 'mhw'
    selected = _resolve_output_vars(output_vars, prefix)

    # Skip the per-event loop when no per-event field was requested.
    metric_names = {f'{prefix}_{f}' for f in _EVENT_METRIC_FIELDS}
    func_kwargs['compute_event_metrics'] = bool(metric_names & set(selected))

    # (seas, thresh, anomaly, is_mhw, duration, category, imax, icum, severity)
    output_dtypes = [np.float32] * 3 + [bool] + [np.float32] * 5

    results = xr.apply_ufunc(
        mhw_1d_wrapper,
        time_ordinal_da,
        ds[temp_var_name],
        kwargs=func_kwargs,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']] * 9,
        vectorize=True,
        dask='parallelized',
        output_dtypes=output_dtypes,
    )
    (seas, thresh, anomaly, is_mhw, duration, category,
     intensity_max, intensity_cum, severity) = results

    available = {
        f'{prefix}_intensity': (anomaly.where(is_mhw), 'intensity'),
        f'{prefix}_severity': (severity.where(is_mhw), 'severity'),
        f'{prefix}_duration': (duration.where(is_mhw), 'duration'),
        f'{prefix}_category': (category.where(is_mhw), 'category'),
        f'{prefix}_max_intensity': (intensity_max.where(is_mhw), 'max_intensity'),
        f'{prefix}_cum_intensity': (intensity_cum.where(is_mhw), 'cum_intensity'),
        'climatology': (seas, 'climatology'),
        f'{prefix}_threshold': (thresh, 'threshold'),
    }

    ds_out = xr.Dataset()
    for name in selected:
        data, attr_key = available[name]
        long_name, units, description = _VAR_ATTRS[attr_key]
        ds_out[name] = data
        ds_out[name].attrs.update(long_name=long_name, units=units,
                                  description=description)

    ds_out = ds_out.assign_coords(ds.coords)
    ds_out.attrs = dict(ds.attrs)
    ds_out.attrs.update({
        'mhw_detection_period': str(tuple(clim_period)),
        'mhw_type': 'Cold Spell' if func_kwargs['cold_spells'] else 'Heatwave',
        'mhw_percentile': func_kwargs['pctile'],
        'mhw_min_duration_days': func_kwargs.get('min_duration', 5),
        'mhw_max_gap_days': func_kwargs.get('max_gap', 2),
        'mhw_join_method': func_kwargs.get('join_method', 'post-filter'),
        'mhw_detection_scope': ('pointwise (each grid cell is an independent '
                                '1-D time series; no spatial object tracking)'),
        'software': f'xrMHW v{__version__} (Hobday et al. 2016 definition)',
    })
    return ds_out
