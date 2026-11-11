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

__all__ = ["detect_mhw_core", "mhw_1d_wrapper", "xrMHW_func", "xrMHW_clim",
           "xrMHW_events", "xrMHW_doy", "xrMHW_detrend", "xrMHW_annual",
           "xrMHW_trend"]

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


def _climatology_1d(year, doy, temp, clim_period, pctile, window_half_width,
                    smooth_pctile, smooth_width):
    """Day-of-year climatology and threshold of one series over its baseline.

    Returns ``(seas_clim_year, thresh_clim_year)``, two arrays of length 366
    on the leap-year day-of-year grid. This is the first half of
    `detect_mhw_core`, unchanged; `xrMHW_clim` calls it on its own.
    """
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
    return seas_clim_year, thresh_clim_year


def _events_1d(temp, clim_seas, clim_thresh, min_duration=5,
               join_across_gaps=True, max_gap=2, cold_spells=False,
               compute_event_metrics=True, join_method='post-filter'):
    """Events of one series against a climatology and threshold given per time step.

    This is the second half of `detect_mhw_core`, unchanged. `clim_seas` and
    `clim_thresh` have the length of `temp`, so the threshold may vary from
    one year to the next (any baseline, not only a fixed one).
    Returns the same 9-tuple as `detect_mhw_core`.
    """
    T = len(temp)
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
    year, doy = _time_metadata(t_ordinal)
    seas_clim_year, thresh_clim_year = _climatology_1d(
        year, doy, temp, clim_period, pctile, window_half_width,
        smooth_pctile, smooth_width)
    return _events_1d(
        temp, seas_clim_year[doy - 1], thresh_clim_year[doy - 1],
        min_duration=min_duration, join_across_gaps=join_across_gaps,
        max_gap=max_gap, cold_spells=cold_spells,
        compute_event_metrics=compute_event_metrics, join_method=join_method)


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
        return _nan_outputs(T)

    # 2. Interpolate only short data gaps.
    _fill_short_gaps(temp, max_gap_interp)

    return detect_mhw_core(
        time_ordinal.astype(int),
        temp.astype(float),
        clim_period=(int(clim_start_year), int(clim_end_year)),
        **kwargs
    )


def _nan_outputs(T):
    """The 9-tuple returned for a cell with no data (land, masked)."""
    nan_arr = np.full(T, np.nan)
    return (nan_arr, nan_arr, nan_arr, np.zeros(T, dtype=bool),
            nan_arr, nan_arr, nan_arr, nan_arr, nan_arr)


def _fill_short_gaps(temp, max_gap_interp):
    """Fill, in place, runs of at most `max_gap_interp` missing values by linear
    interpolation. Longer gaps stay NaN."""
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
    return temp


def _clim_1d_wrapper(time_ordinal, temp, clim_start_year, clim_end_year,
                     max_gap_interp=2, pctile=90, window_half_width=5,
                     smooth_pctile=True, smooth_width=31):
    """Climatology and threshold of one cell, on the 366-day grid.

    Same NaN handling as `mhw_1d_wrapper`, so `xrMHW_clim` followed by
    `xrMHW_events` reproduces `xrMHW_func` exactly.
    """
    temp = temp.copy()  # Dask hands over read-only buffers
    if np.isnan(temp).all():
        nan366 = np.full(_NDOY, np.nan)
        return nan366, nan366.copy()
    _fill_short_gaps(temp, max_gap_interp)
    t_ordinal = time_ordinal.astype(int)
    year, doy = _time_metadata(t_ordinal)
    return _climatology_1d(
        year, doy, temp.astype(float), (int(clim_start_year), int(clim_end_year)),
        pctile, window_half_width, smooth_pctile, smooth_width)


_DAYS_PER_DECADE = 3652.5


def _expand_clim(time_ordinal, seas, thresh, coef=None, layout='doy',
                 period_starts=None, t0=0.0, offset_to='both'):
    """Climatology and threshold of one cell at every time step.

    layout 'doy': (366,) arrays indexed by day of year; 'period': (n, 366)
    arrays, one row per period starting in `period_starts`; 'time': already
    one value per time step. `coef` adds a polynomial offset in decades since
    `t0` (detrended and adaptation-adjusted baselines), to both fields or only
    to the threshold.
    """
    if layout == 'time':
        seas_t, thresh_t = np.asarray(seas, float), np.asarray(thresh, float)
    else:
        year, doy = _time_metadata(time_ordinal)
        if layout == 'period':
            row = np.searchsorted(np.asarray(period_starts), year, side='right') - 1
            seas_t, thresh_t = seas[row, doy - 1], thresh[row, doy - 1]
        else:
            seas_t, thresh_t = seas[doy - 1], thresh[doy - 1]
        seas_t, thresh_t = np.asarray(seas_t, float), np.asarray(thresh_t, float)
    if coef is not None:
        offset = np.polyval(np.asarray(coef, float),
                            (time_ordinal - t0) / _DAYS_PER_DECADE)
        thresh_t = thresh_t + offset
        if offset_to == 'both':
            seas_t = seas_t + offset
    return seas_t, thresh_t


def _events_1d_wrapper(time_ordinal, temp, seas, thresh, *coef, layout='doy',
                       period_starts=None, t0=0.0, offset_to='both',
                       max_gap_interp=2, **kwargs):
    """Events of one cell against a given climatology and threshold (any layout
    of `_expand_clim`). Same NaN handling as `mhw_1d_wrapper`."""
    T = len(time_ordinal)
    temp = temp.copy()
    if np.isnan(temp).all():
        return _nan_outputs(T)
    _fill_short_gaps(temp, max_gap_interp)
    seas_t, thresh_t = _expand_clim(
        time_ordinal.astype(int), seas, thresh, coef[0] if coef else None,
        layout, period_starts, t0, offset_to)
    return _events_1d(temp.astype(float), seas_t, thresh_t, **kwargs)


def _trend_1d(t_ordinal, temp, year, doy, clim_period, degree, t0,
              window_half_width, smooth_width, smooth_pctile):
    """Polynomial trend of the anomaly of one series, in decades since `t0`.

    The anomaly is taken from the fixed climatology of the raw series over
    `clim_period`; the trend is fitted over the whole record and shifted to
    have zero mean over `clim_period`, so removing it leaves the reference
    period at its observed level (Smith et al. 2025, Fig. 4).
    Returns (coef, offset per time step).
    """
    seas, _ = _climatology_1d(year, doy, temp, clim_period, 50,
                              window_half_width, smooth_pctile, smooth_width)
    x = (t_ordinal - t0) / _DAYS_PER_DECADE
    anom = temp - seas[doy - 1]
    ok = np.isfinite(anom)
    if ok.sum() <= degree:
        coef = np.full(degree + 1, np.nan)
        return coef, np.full(len(temp), np.nan)
    coef = np.polyfit(x[ok], anom[ok], degree)
    in_ref = (year >= clim_period[0]) & (year <= clim_period[1])
    coef[-1] -= np.polyval(coef, x[in_ref]).mean()
    return coef, np.polyval(coef, x)


def _detrended_clim_1d_wrapper(time_ordinal, temp, clim_start_year, clim_end_year,
                               degree=1, t0=0.0, max_gap_interp=2, pctile=90,
                               window_half_width=5, smooth_pctile=True,
                               smooth_width=31):
    """Fixed climatology of the detrended series, and the trend coefficients."""
    temp = temp.copy()
    if np.isnan(temp).all():
        return np.full(_NDOY, np.nan), np.full(_NDOY, np.nan), np.full(degree + 1, np.nan)
    _fill_short_gaps(temp, max_gap_interp)
    t_ordinal = time_ordinal.astype(int)
    year, doy = _time_metadata(t_ordinal)
    temp = temp.astype(float)
    period = (int(clim_start_year), int(clim_end_year))
    coef, offset = _trend_1d(t_ordinal, temp, year, doy, period, degree, t0,
                             window_half_width, smooth_width, smooth_pctile)
    seas, thresh = _climatology_1d(year, doy, temp - offset, period, pctile,
                                   window_half_width, smooth_pctile, smooth_width)
    return seas, thresh, coef


def _trend_coef_1d_wrapper(time_ordinal, temp, clim_start_year, clim_end_year,
                           degree=1, t0=0.0, max_gap_interp=2, window_half_width=5,
                           smooth_pctile=True, smooth_width=31):
    """Trend coefficients only (for `xrMHW_detrend`)."""
    temp = temp.copy()
    if np.isnan(temp).all():
        return np.full(degree + 1, np.nan)
    _fill_short_gaps(temp, max_gap_interp)
    t_ordinal = time_ordinal.astype(int)
    year, doy = _time_metadata(t_ordinal)
    coef, _ = _trend_1d(t_ordinal, temp.astype(float), year, doy,
                        (int(clim_start_year), int(clim_end_year)), degree, t0,
                        window_half_width, smooth_width, smooth_pctile)
    return coef


def _polyval_1d(time_ordinal, coef, t0=0.0):
    return np.polyval(coef, (time_ordinal.astype(float) - t0) / _DAYS_PER_DECADE)


def _periodic_clim_1d_wrapper(time_ordinal, temp, windows=(), max_gap_interp=2,
                              pctile=90, window_half_width=5, smooth_pctile=True,
                              smooth_width=31):
    """One fixed climatology per period; `windows` holds each period's baseline
    years. Identical windows are computed once."""
    n = len(windows)
    temp = temp.copy()
    if np.isnan(temp).all():
        return np.full((n, _NDOY), np.nan), np.full((n, _NDOY), np.nan)
    _fill_short_gaps(temp, max_gap_interp)
    year, doy = _time_metadata(time_ordinal.astype(int))
    return _window_climatologies(year, doy, temp.astype(float), windows, pctile,
                                 window_half_width, smooth_pctile, smooth_width)


def _window_climatologies(year, doy, temp, windows, pctile, window_half_width,
                          smooth_pctile, smooth_width):
    """One fixed climatology per baseline window (n, 366); identical windows
    are computed once."""
    n = len(windows)
    seas, thresh = np.empty((n, _NDOY)), np.empty((n, _NDOY))
    done = {}
    for i, w in enumerate(windows):
        w = (int(w[0]), int(w[1]))
        if w not in done:
            done[w] = _climatology_1d(year, doy, temp, w, pctile, window_half_width,
                                      smooth_pctile, smooth_width)
        seas[i], thresh[i] = done[w]
    return seas, thresh


def _shifting_clim_1d_wrapper(time_ordinal, temp, windows=(), degree=1, t0=0.0,
                              max_gap_interp=2, pctile=90, window_half_width=5,
                              smooth_pctile=True, smooth_width=31):
    """Shifting baseline with the trend removed first (Smith et al. 2025,
    recommendation 3e): the polynomial trend of the anomaly over the whole
    record is removed, one climatology per year is computed from its window,
    and the trend coefficients are returned so that climatology and
    threshold follow the trend again when applied to the raw series."""
    n = len(windows)
    temp = temp.copy()
    if np.isnan(temp).all():
        return (np.full((n, _NDOY), np.nan), np.full((n, _NDOY), np.nan),
                np.full(degree + 1, np.nan))
    _fill_short_gaps(temp, max_gap_interp)
    t_ordinal = time_ordinal.astype(int)
    year, doy = _time_metadata(t_ordinal)
    temp = temp.astype(float)
    coef, offset = _trend_1d(t_ordinal, temp, year, doy, (int(year[0]), int(year[-1])),
                             degree, t0, window_half_width, smooth_width, smooth_pctile)
    seas, thresh = _window_climatologies(year, doy, temp - offset, windows, pctile,
                                         window_half_width, smooth_pctile, smooth_width)
    return seas, thresh, coef


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


def _prepare_input(ds, temp_var_name):
    """Check the input dataset and put the whole time axis in one chunk."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError("`ds` must be an xarray.Dataset "
                        "(use da.to_dataset() for a DataArray).")
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
                UserWarning, stacklevel=3)
            ds = ds.chunk({'time': -1})
    return ds


def _time_ordinal_da(ds):
    """Ordinal day numbers of the time coordinate, as a DataArray on 'time'."""
    time_index = ds.indexes['time']
    try:
        time_ordinal_np = time_index.map(lambda x: x.toordinal()).values
    except AttributeError:
        raise TypeError("Time coordinate must be convertible to ordinal "
                        "(datetime-like).")
    return xr.DataArray(time_ordinal_np, coords={'time': ds['time']}, dims='time')


def _assemble_outputs(results, ds, prefix, selected):
    """Mask the per-cell results with the event mask and build the output dataset."""
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
    return ds_out


def xrMHW_func(ds, temp_var_name, clim_period, output_vars=None,
               baseline='fixed', **kwargs):
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
    baseline : {'fixed', 'detrended', 'adaptation', 'periodic', 'shifting'}
        Default 'fixed'. Any other baseline (Smith et al. 2025) runs
        `xrMHW_clim` and `xrMHW_events` in turn; its options (``trend``,
        ``degree``, ``rate``, ``ref_length``, ``update_every``) go in
        `kwargs` (also ``window``, ``align`` and ``detrend_first`` for
        'shifting'). See `xrMHW_clim`.

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
    if baseline != 'fixed':
        if baseline not in _BASELINES:
            raise ValueError(f"baseline must be one of {sorted(_BASELINES)}, "
                             f"not {baseline!r}.")
        clim_keys = set(_CLIM_DEFAULTS) | set(_BASELINES[baseline]) | {'cold_spells'}
        clim_kw = {k: v for k, v in kwargs.items() if k in clim_keys}
        event_kw = {k: v for k, v in kwargs.items() if k not in clim_keys}
        if 'max_gap_interp' in clim_kw:
            event_kw['max_gap_interp'] = clim_kw['max_gap_interp']
        clim = xrMHW_clim(ds, temp_var_name, clim_period, baseline=baseline, **clim_kw)
        prefix = 'mcs' if clim_kw.get('cold_spells') else 'mhw'
        return xrMHW_events(ds, temp_var_name, clim,
                            output_vars=(_canonical_outputs(prefix)
                                         if output_vars is None else output_vars),
                            **event_kw)

    ds = _prepare_input(ds, temp_var_name)
    time_ordinal_da = _time_ordinal_da(ds)

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
    ds_out = _assemble_outputs(results, ds, prefix, selected)
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


# =============================================================================
# 4. MODULAR API: climatology and events as separate steps
# =============================================================================

_CLIM_DEFAULTS = dict(pctile=None, window_half_width=5, smooth_pctile=True,
                      smooth_width=31, max_gap_interp=2)
_EVENT_DEFAULTS = dict(min_duration=5, join_across_gaps=True, max_gap=2,
                       join_method='post-filter')
# Baselines of Smith et al. (2025) and their own parameters (defaults).
_BASELINES = {
    'fixed': {},
    'detrended': {'trend': 'linear', 'degree': None},
    'adaptation': {'rate': None},
    'periodic': {'ref_length': 30, 'update_every': 10},
    'shifting': {'window': 30, 'align': 'lagged', 'detrend_first': True,
                 'trend': 'linear', 'degree': None},
}


def _check_kwargs(kwargs, allowed, func):
    unknown = sorted(set(kwargs) - set(allowed))
    if unknown:
        raise TypeError(f"{func}() got unexpected keyword argument(s) {unknown}. "
                        f"Allowed: {sorted(allowed)}.")


def xrMHW_doy(time):
    """Day of year on xrMHW's 366-day (leap-year) grid.

    Every year, leap or not, is aligned to the leap-year grid, so 29 February
    is always day 60 and 1 March always day 61, as in the reference
    implementation. (The ``dayofyear`` of pandas and xarray gives 60 to
    1 March in non-leap years.) Use it to index a climatology from
    `xrMHW_clim` by date, e.g. ``clim.sel(doy=xrMHW_doy(ds.time))``.

    Parameters
    ----------
    time : xr.DataArray, pandas.DatetimeIndex or array-like of datetimes

    Returns
    -------
    xr.DataArray of int16, named 'doy', with the input 'time' coordinate when
    `time` is an xarray object.
    """
    if isinstance(time, xr.DataArray):
        values, coords, dims = time.values, {time.dims[0]: time}, time.dims
    else:
        values, coords, dims = time, None, ('time',)
    dates = pd.DatetimeIndex(values)
    dmap = _leap_doy_map()
    doy = np.array([dmap[(m, d)] for m, d in zip(dates.month, dates.day)],
                   dtype=np.int16)
    da = xr.DataArray(doy, coords=coords, dims=dims, name='doy')
    da.attrs.update(long_name='day of year on a 366-day (leap-year) grid',
                    units='1', description='29 February is day 60; 1 March is '
                    'always day 61.')
    return da


def _centre_ordinal(start_year, end_year):
    return 0.5 * (date(int(start_year), 1, 1).toordinal()
                  + date(int(end_year), 12, 31).toordinal())


def _trend_degree(trend, degree):
    if trend == 'linear':
        if degree not in (None, 1):
            raise ValueError("trend='linear' has degree 1; use "
                             "trend='polynomial' for another degree.")
        return 1
    if trend == 'polynomial':
        if degree is None or int(degree) < 1:
            raise ValueError("trend='polynomial' needs degree >= 1, e.g. degree=3.")
        return int(degree)
    raise ValueError(f"trend must be 'linear' or 'polynomial', not {trend!r}.")


def _shifting_windows(years, window, align):
    """Baseline window of every year of the record. 'lagged': the `window`
    years before; 'centred': the `window` years around it. A window that leaves
    the record is moved to the nearest complete one (the first, or for
    'centred' near the end, the last), which makes those years a fixed
    baseline (an approximation)."""
    if align not in ('lagged', 'centred'):
        raise ValueError(f"align must be 'lagged' or 'centred', not {align!r}.")
    y0, y1 = int(years.min()), int(years.max())
    if y1 - y0 + 1 < window:
        raise ValueError(f"The record ({y0}-{y1}) is shorter than window={window} years.")
    starts, windows, approx = list(range(y0, y1 + 1)), [], []
    for y in starts:
        a = y - window if align == 'lagged' else y - window // 2
        w = (a, a + window - 1)
        if w[0] < y0:
            w = (y0, y0 + window - 1)
        elif w[1] > y1:
            w = (y1 - window + 1, y1)
        if w != (a, a + window - 1):
            approx.append(y)
        windows.append(w)
    return starts, windows, approx


def _year_ranges(years):
    """Compact text for a list of years, e.g. '1961-1990, 2010-2020'."""
    out, run = [], [years[0]]
    for y in years[1:]:
        if y == run[-1] + 1:
            run.append(y)
        else:
            out.append(run)
            run = [y]
    out.append(run)
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else f"{r[0]}" for r in out)


def _periodic_windows(years, ref_length, update_every):
    """Period start years and the baseline of each period (WMO practice: the
    reference period is the `ref_length` years before the period). A period
    whose baseline starts before the record uses the first complete window."""
    y0, y1 = int(years.min()), int(years.max())
    if y1 - y0 + 1 < ref_length:
        raise ValueError(f"The record ({y0}-{y1}) is shorter than ref_length="
                         f"{ref_length} years.")
    first = y0 - ((y0 - 1) % update_every)          # e.g. 1991, 2001, ... for 10
    starts = list(range(first, y1 + 1, update_every))
    windows, approx = [], []
    for s in starts:
        w = (s - ref_length, s - 1)
        if w[0] < y0:
            w = (y0, y0 + ref_length - 1)
            approx.append(s)
        windows.append(w)
    return starts, windows, approx


def xrMHW_clim(ds, temp_var_name, clim_period=None, cold_spells=False,
               baseline='fixed', **kwargs):
    """
    Seasonal climatology and percentile threshold, without detecting events.

    This is the first half of `xrMHW_func`, on its own. Its output can be
    stored, inspected, reused for several detections, or applied to another
    dataset or period with `xrMHW_events`. With the fixed baseline,
    ``xrMHW_events(ds, var, xrMHW_clim(ds, var, clim_period))`` gives exactly
    the same events as ``xrMHW_func(ds, var, clim_period)``.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a temperature variable and a 'time' coordinate.
    temp_var_name : str
        Name of the temperature variable in `ds`.
    clim_period : tuple of (int, int)
        Start and end years (inclusive) of the reference period, e.g.
        ``(1991, 2020)``. Required, except for the periodic and shifting
        baselines.
    cold_spells : bool, default False
        Compute the threshold for marine cold spells (10th percentile by
        default) instead of heatwaves (90th).
    baseline : {'fixed', 'detrended', 'adaptation', 'periodic', 'shifting'}
        How the threshold relates to time (Smith et al. 2025, Prog. Oceanogr.
        231, 103404):

        * ``'fixed'`` (default) — one reference period for the whole record.
        * ``'detrended'`` — the long-term trend of the anomaly is fitted over
          the whole record and removed before the fixed climatology is
          computed; climatology and threshold then follow the trend. Options
          ``trend='linear'`` (default) or ``'polynomial'`` with ``degree``.
          A wrong trend shape gives spurious events (Smith et al., Box 1).
        * ``'adaptation'`` — fixed threshold rising at ``rate`` degC per
          decade from the centre of the reference period, for an organism
          adapting at that rate. Only the threshold moves; the climatology
          stays fixed.
        * ``'periodic'`` — the reference period is the ``ref_length`` years
          (default 30) before each period of ``update_every`` years (default
          10), as recommended by the WMO: e.g. 2021-2030 uses 1991-2020.
          Periods without a complete reference period before them use the
          first complete one in the record, with a warning.
        * ``'shifting'`` — the reference period moves every year: with
          ``align='lagged'`` (default) it is the ``window`` years (default 30)
          before the year, as in Smith et al.'s examples; ``'centred'`` uses the
          years around it. Years without a complete window use the nearest
          complete one in the record, with a warning. With
          ``detrend_first=True`` (default; Smith et al., recommendation 3e) the
          trend of the anomaly over the whole record (``trend``, ``degree``) is
          removed first, so that warming is not read as growing variability.
    **kwargs
        * **pctile** (int/float, default 90, or 10 for cold spells).
        * **window_half_width** (int, default 5) — the window is ``2*w + 1`` days.
        * **smooth_pctile** (bool, default True) and **smooth_width**
          (int, default 31) — smoothing of climatology and threshold.
        * **max_gap_interp** (int, default 2) — longest run of missing values
          filled by interpolation before the climatology is computed.
        * the options of the chosen baseline, listed above.

    Returns
    -------
    xr.Dataset
        ``climatology`` and ``mhw_threshold`` (``mcs_threshold`` for cold
        spells), float64, with a ``doy`` dimension of length 366 (see
        `xrMHW_doy`) in place of ``time`` (plus ``period`` for the periodic
        and shifting baselines, and ``offset_coef`` when a trend or a rate
        moves the threshold).
        The attributes record the baseline and every parameter;
        `xrMHW_events` reads them.
    """
    if baseline not in _BASELINES:
        raise ValueError(f"baseline must be one of {sorted(_BASELINES)}, "
                         f"not {baseline!r}.")
    _check_kwargs(kwargs, set(_CLIM_DEFAULTS) | set(_BASELINES[baseline]),
                  'xrMHW_clim')
    opts = {k: kwargs.pop(k, v) for k, v in _BASELINES[baseline].items()}
    params = dict(_CLIM_DEFAULTS, **kwargs)
    if params['pctile'] is None:
        params['pctile'] = 10 if cold_spells else 90
    if baseline not in ('periodic', 'shifting') and clim_period is None:
        raise ValueError(f"baseline={baseline!r} needs clim_period=(start, end).")
    prefix = 'mcs' if cold_spells else 'mhw'

    ds = _prepare_input(ds, temp_var_name)
    time_ordinal_da = _time_ordinal_da(ds)
    da = ds[temp_var_name]
    common = dict(input_core_dims=[['time'], ['time']], vectorize=True,
                  dask='parallelized')
    attrs = {'xrMHW_baseline': baseline}
    coef = None

    if baseline in ('fixed', 'adaptation'):
        seas, thresh = xr.apply_ufunc(
            _clim_1d_wrapper, time_ordinal_da, da,
            kwargs={'clim_start_year': clim_period[0],
                    'clim_end_year': clim_period[1], **params},
            output_core_dims=[['doy'], ['doy']],
            output_dtypes=[np.float64, np.float64],
            dask_gufunc_kwargs={'output_sizes': {'doy': _NDOY}}, **common)
        if baseline == 'adaptation':
            if opts['rate'] is None:
                raise ValueError("baseline='adaptation' needs rate=<degC per decade>.")
            t0 = _centre_ordinal(*clim_period)
            rate = float(opts['rate'])
            coef = xr.DataArray(np.array([rate, 0.0]), dims='coef')
            coef = coef.broadcast_like(seas.isel(doy=0, drop=True)).copy()
            attrs.update(xrMHW_offset_t0=t0, xrMHW_offset_to='threshold',
                         xrMHW_rate_per_decade=rate,
                         xrMHW_t_ref=(str(date.fromordinal(int(t0)))
                                      + (' 12:00' if t0 % 1 else '')))

    elif baseline == 'detrended':
        degree = _trend_degree(opts['trend'], opts['degree'])
        t_ord = time_ordinal_da.values
        t0 = 0.5 * (float(t_ord[0]) + float(t_ord[-1]))
        seas, thresh, coef = xr.apply_ufunc(
            _detrended_clim_1d_wrapper, time_ordinal_da, da,
            kwargs={'clim_start_year': clim_period[0],
                    'clim_end_year': clim_period[1], 'degree': degree, 't0': t0,
                    **params},
            output_core_dims=[['doy'], ['doy'], ['coef']],
            output_dtypes=[np.float64] * 3,
            dask_gufunc_kwargs={'output_sizes': {'doy': _NDOY, 'coef': degree + 1}},
            **common)
        attrs.update(xrMHW_offset_t0=t0, xrMHW_offset_to='both',
                     xrMHW_trend=opts['trend'], xrMHW_trend_degree=degree)

    elif baseline == 'shifting':
        years = np.asarray(ds.indexes['time'].year)
        starts, windows, approx = _shifting_windows(years, int(opts['window']),
                                                    opts['align'])
        if approx:
            warnings.warn(
                f"No complete {opts['window']}-year {opts['align']} window for the "
                f"years {_year_ranges(approx)}; they use the nearest complete "
                "window in the record, which makes them a fixed baseline "
                "(an approximation).", UserWarning, stacklevel=2)
        sizes = {'period': len(starts), 'doy': _NDOY}
        if opts['detrend_first']:
            degree = _trend_degree(opts['trend'], opts['degree'])
            t_ord = time_ordinal_da.values
            t0 = 0.5 * (float(t_ord[0]) + float(t_ord[-1]))
            seas, thresh, coef = xr.apply_ufunc(
                _shifting_clim_1d_wrapper, time_ordinal_da, da,
                kwargs={'windows': tuple(windows), 'degree': degree, 't0': t0,
                        **params},
                output_core_dims=[['period', 'doy'], ['period', 'doy'], ['coef']],
                output_dtypes=[np.float64] * 3,
                dask_gufunc_kwargs={'output_sizes': dict(sizes, coef=degree + 1)},
                **common)
            attrs.update(xrMHW_offset_t0=t0, xrMHW_offset_to='both',
                         xrMHW_trend=opts['trend'], xrMHW_trend_degree=degree)
        else:
            seas, thresh = xr.apply_ufunc(
                _periodic_clim_1d_wrapper, time_ordinal_da, da,
                kwargs={'windows': tuple(windows), **params},
                output_core_dims=[['period', 'doy'], ['period', 'doy']],
                output_dtypes=[np.float64, np.float64],
                dask_gufunc_kwargs={'output_sizes': sizes}, **common)
        seas = seas.assign_coords(period=np.asarray(starts, dtype=np.int32))
        thresh = thresh.assign_coords(period=np.asarray(starts, dtype=np.int32))
        attrs.update(xrMHW_window=int(opts['window']), xrMHW_align=opts['align'],
                     xrMHW_detrend_first=int(bool(opts['detrend_first'])),
                     xrMHW_period_baselines=str(dict(zip(starts, windows))))

    else:  # periodic
        years = ds.indexes['time'].year
        starts, windows, approx = _periodic_windows(
            np.asarray(years), int(opts['ref_length']), int(opts['update_every']))
        if approx:
            warnings.warn(
                f"No complete {opts['ref_length']}-year reference period before "
                f"the periods starting in {approx}; they use the first complete "
                f"one, {windows[starts.index(approx[0])]}, which makes them a "
                "fixed baseline (an approximation).", UserWarning, stacklevel=2)
        seas, thresh = xr.apply_ufunc(
            _periodic_clim_1d_wrapper, time_ordinal_da, da,
            kwargs={'windows': tuple(windows), **params},
            output_core_dims=[['period', 'doy'], ['period', 'doy']],
            output_dtypes=[np.float64, np.float64],
            dask_gufunc_kwargs={'output_sizes': {'period': len(starts),
                                                 'doy': _NDOY}}, **common)
        seas = seas.assign_coords(period=np.asarray(starts, dtype=np.int32))
        thresh = thresh.assign_coords(period=np.asarray(starts, dtype=np.int32))
        attrs.update(xrMHW_ref_length=int(opts['ref_length']),
                     xrMHW_update_every=int(opts['update_every']),
                     xrMHW_period_baselines=str(dict(zip(starts, windows))))

    out = xr.Dataset({'climatology': seas, f'{prefix}_threshold': thresh})
    if coef is not None:
        out['offset_coef'] = coef
        out['offset_coef'].attrs.update(
            long_name='polynomial offset coefficients',
            units='degC per decade**k (highest power first)',
            description='Offset added at each date to the threshold (and, '
            "when xrMHW_offset_to == 'both', to the climatology): "
            'numpy.polyval(offset_coef, (t - xrMHW_offset_t0) / 3652.5), '
            't in ordinal days.')
    out = out.assign_coords(doy=np.arange(1, _NDOY + 1, dtype=np.int16))
    out['doy'].attrs.update(long_name='day of year on a 366-day (leap-year) grid',
                            units='1',
                            description='29 February is day 60; 1 March is '
                            'always day 61 (see xrMHW_doy).')
    if 'period' in out.dims:
        out['period'].attrs.update(long_name='first year of the period',
                                   units='year')
    for name, key in (('climatology', 'climatology'),
                      (f'{prefix}_threshold', 'threshold')):
        long_name, units, description = _VAR_ATTRS[key]
        out[name].attrs.update(long_name=long_name, units=units,
                               description=description)
    # Keep the coordinates that do not depend on time (lat, lon, depth, ...).
    out = out.assign_coords({k: v for k, v in ds.coords.items()
                             if 'time' not in v.dims and k not in out.coords})
    out.attrs.update(attrs)
    out.attrs.update({
        'xrMHW_clim_period': (str(tuple(clim_period)) if clim_period is not None
                              else 'per period, see xrMHW_period_baselines'),
        'xrMHW_pctile': params['pctile'],
        'xrMHW_window_half_width': params['window_half_width'],
        'xrMHW_smooth_pctile': int(bool(params['smooth_pctile'])),
        'xrMHW_smooth_width': params['smooth_width'],
        'xrMHW_max_gap_interp': params['max_gap_interp'],
        'xrMHW_cold_spells': int(bool(cold_spells)),
        'software': f'xrMHW v{__version__} (Hobday et al. 2016 definition)',
    })
    return out


def xrMHW_detrend(ds, temp_var_name, clim_period, trend='linear', degree=None,
                  **kwargs):
    """
    Remove the long-term trend from a temperature field, cell by cell.

    The trend is a polynomial fitted over the whole record to the anomaly from
    the fixed climatology of `clim_period`, and shifted to have zero mean over
    `clim_period`, so the reference period keeps its observed level (as in
    Smith et al. 2025, Fig. 4). ``baseline='detrended'`` in `xrMHW_clim` uses
    the same trend; detecting events in the returned field against a fixed
    climatology gives the same events.

    Parameters
    ----------
    ds, temp_var_name, clim_period : as in `xrMHW_clim`
    trend : {'linear', 'polynomial'}, default 'linear'
    degree : int, required for ``trend='polynomial'``
    **kwargs
        **window_half_width**, **smooth_pctile**, **smooth_width** and
        **max_gap_interp**, for the climatology used to form the anomaly.

    Returns
    -------
    xr.Dataset
        The detrended temperature (lazy if `ds` is) and ``trend_coef``,
        the polynomial coefficients in degC per decade**k.
    """
    allowed = {'window_half_width', 'smooth_pctile', 'smooth_width', 'max_gap_interp'}
    _check_kwargs(kwargs, allowed, 'xrMHW_detrend')
    params = {k: v for k, v in _CLIM_DEFAULTS.items() if k in allowed}
    params.update(kwargs)
    degree = _trend_degree(trend, degree)
    ds = _prepare_input(ds, temp_var_name)
    tord = _time_ordinal_da(ds)
    t0 = 0.5 * (float(tord.values[0]) + float(tord.values[-1]))
    coef = xr.apply_ufunc(
        _trend_coef_1d_wrapper, tord, ds[temp_var_name],
        kwargs={'clim_start_year': clim_period[0], 'clim_end_year': clim_period[1],
                'degree': degree, 't0': t0, **params},
        input_core_dims=[['time'], ['time']], output_core_dims=[['coef']],
        vectorize=True, dask='parallelized', output_dtypes=[np.float64],
        dask_gufunc_kwargs={'output_sizes': {'coef': degree + 1}})
    offset = xr.apply_ufunc(
        _polyval_1d, tord, coef, kwargs={'t0': t0},
        input_core_dims=[['time'], ['coef']], output_core_dims=[['time']],
        vectorize=True, dask='parallelized', output_dtypes=[np.float64])
    detrended = (ds[temp_var_name] - offset).transpose(*ds[temp_var_name].dims)
    detrended.attrs = dict(ds[temp_var_name].attrs)
    out = xr.Dataset({temp_var_name: detrended, 'trend_coef': coef})
    out.attrs.update({'xrMHW_trend': trend, 'xrMHW_trend_degree': degree,
                      'xrMHW_offset_t0': t0,
                      'xrMHW_clim_period': str(tuple(clim_period))})
    return out


def _check_clim(clim, da):
    """Validate a climatology for `xrMHW_events`; return (threshold name, layout)."""
    if not isinstance(clim, xr.Dataset):
        raise TypeError("`clim` must be an xarray.Dataset, as returned by "
                        "xrMHW_clim, with 'climatology' and 'mhw_threshold' "
                        "(or 'mcs_threshold').")
    thr = [v for v in ('mhw_threshold', 'mcs_threshold') if v in clim]
    if 'climatology' not in clim or len(thr) != 1:
        raise ValueError("`clim` must contain 'climatology' and exactly one of "
                         "'mhw_threshold' or 'mcs_threshold'; found "
                         f"{list(clim.data_vars)}.")
    thr = thr[0]
    dims = set(clim[thr].dims) | set(clim['climatology'].dims)
    core = [d for d in ('doy', 'time') if d in dims]
    if len(core) != 1:
        raise ValueError("The climatology must vary along exactly one of 'doy' "
                         "(a seasonal cycle, as from xrMHW_clim) or 'time' "
                         f"(one value per date); it has dims {sorted(dims)}.")
    layout = core[0]
    if layout == 'doy' and clim.sizes['doy'] != _NDOY:
        raise ValueError(f"'doy' must have length {_NDOY} (the 366-day grid of "
                         f"xrMHW_doy); it has {clim.sizes['doy']}.")
    if 'period' in dims:
        if layout != 'doy' or 'period' not in clim.coords:
            raise ValueError("A climatology with a 'period' dimension needs 'doy' "
                             "and a 'period' coordinate holding the first year "
                             "of each period.")
        years = da.indexes['time'].year if 'time' in da.indexes else None
        if years is not None and int(years.min()) < int(clim['period'].min()):
            raise ValueError("The first period starts after the first year of "
                             "the temperature record.")
        layout = 'period'
    if 'offset_coef' in clim and layout == 'time':
        raise ValueError("'offset_coef' only applies to a climatology on 'doy'.")
    if layout == 'time':
        t_clim = clim['time'].values if 'time' in clim.coords else None
        if (clim.sizes['time'] != da.sizes['time']
                or (t_clim is not None and not np.array_equal(t_clim, da['time'].values))):
            raise ValueError("A climatology given per date must have the same "
                             "'time' coordinate as the temperature.")
    grid_dims = dims | (set(clim['offset_coef'].dims) if 'offset_coef' in clim else set())
    for d in grid_dims - {'doy', 'time', 'period', 'coef'}:
        if d not in da.dims:
            raise ValueError(f"The climatology has dimension '{d}', which the "
                             f"temperature does not have (its dims: {da.dims}).")
        if clim.sizes[d] != da.sizes[d]:
            raise ValueError(f"Dimension '{d}' has size {clim.sizes[d]} in the "
                             f"climatology and {da.sizes[d]} in the temperature.")
        if d in clim.coords and d in da.coords and not np.array_equal(
                clim[d].values, da[d].values):
            raise ValueError(f"Coordinate '{d}' differs between the climatology "
                             "and the temperature; they must be on the same grid.")
    return thr, layout


def xrMHW_events(ds, temp_var_name, clim, output_vars=None, **kwargs):
    """
    Detect marine heatwaves (or cold spells) against a given climatology and
    threshold.

    This is the second half of `xrMHW_func`, on its own. The climatology can
    come from `xrMHW_clim` on the same data (with the fixed baseline the
    result is then identical to `xrMHW_func`), from another period or dataset,
    or be built by hand: any threshold that varies with the day of year
    (dimension ``doy``, length 366, see `xrMHW_doy`) or with the date
    (dimension ``time``, same as `ds`).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a temperature variable and a 'time' coordinate.
    temp_var_name : str
        Name of the temperature variable in `ds`.
    clim : xr.Dataset
        ``climatology`` and ``mhw_threshold`` (heatwaves) or ``mcs_threshold``
        (cold spells), on the grid of `ds`, as returned by `xrMHW_clim` with
        any baseline. Which threshold is present decides what is detected.
    output_vars : str or sequence of str, optional
        Which variables to return. ``None`` (default) returns the six event
        fields; ``'climatology'`` and ``'threshold'`` can be added, expanded
        to every date.
    **kwargs
        * **min_duration** (int, default 5) — minimum event length in days.
        * **join_across_gaps** (bool, default True) and **max_gap**
          (int, default 2) — joining of events separated by a brief return to
          normal conditions.
        * **join_method** ({'post-filter', 'pre-filter'}, default
          'post-filter') — order of joining relative to the duration filter.
        * **max_gap_interp** (int) — longest run of missing values filled by
          interpolation; by default the value used for `clim`, or 2.

    Returns
    -------
    xr.Dataset
        The same event fields as `xrMHW_func`, on the input grid. Intensity
        and severity are measured from the climatology of the chosen baseline.
    """
    _check_kwargs(kwargs, set(_EVENT_DEFAULTS) | {'max_gap_interp'}, 'xrMHW_events')
    ds = _prepare_input(ds, temp_var_name)
    da = ds[temp_var_name]
    thr_name, layout = _check_clim(clim, da)
    cold_spells = thr_name == 'mcs_threshold'
    prefix = 'mcs' if cold_spells else 'mhw'

    params = dict(_EVENT_DEFAULTS, **kwargs)
    max_gap_interp = params.pop('max_gap_interp',
                                clim.attrs.get('xrMHW_max_gap_interp', 2))

    if output_vars is None:
        selected = _canonical_outputs(prefix)[:6]
    else:
        selected = _resolve_output_vars(output_vars, prefix)
    metric_names = {f'{prefix}_{f}' for f in _EVENT_METRIC_FIELDS}

    seas, thresh = clim['climatology'], clim[thr_name]
    core = {'doy': ['doy'], 'period': ['period', 'doy'], 'time': ['time']}[layout]
    if layout == 'time':
        seas = seas.assign_coords(time=da['time'])
        thresh = thresh.assign_coords(time=da['time'])
        if seas.chunks is not None:
            seas, thresh = seas.chunk({'time': -1}), thresh.chunk({'time': -1})
    inputs = [_time_ordinal_da(ds), da, seas, thresh]
    core_dims = [['time'], ['time'], core, core]
    extra = dict(layout=layout)
    if layout == 'period':
        extra['period_starts'] = tuple(int(p) for p in clim['period'].values)
    if 'offset_coef' in clim:
        inputs.append(clim['offset_coef'])
        core_dims.append(['coef'])
        extra.update(t0=float(clim.attrs.get('xrMHW_offset_t0', 0.0)),
                     offset_to=clim.attrs.get('xrMHW_offset_to', 'both'))

    results = xr.apply_ufunc(
        _events_1d_wrapper, *inputs,
        kwargs=dict(max_gap_interp=int(max_gap_interp), cold_spells=cold_spells,
                    compute_event_metrics=bool(metric_names & set(selected)),
                    **extra, **params),
        input_core_dims=core_dims,
        output_core_dims=[['time']] * 9,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32] * 3 + [bool] + [np.float32] * 5,
    )

    ds_out = _assemble_outputs(results, ds, prefix, selected)
    ds_out.attrs.update({
        'mhw_detection_period': clim.attrs.get('xrMHW_clim_period', 'not recorded'),
        'mhw_baseline': clim.attrs.get('xrMHW_baseline', 'user-supplied'),
        'mhw_type': 'Cold Spell' if cold_spells else 'Heatwave',
        'mhw_percentile': clim.attrs.get('xrMHW_pctile', 'not recorded'),
        'mhw_min_duration_days': params['min_duration'],
        'mhw_max_gap_days': params['max_gap'],
        'mhw_join_method': params['join_method'],
        'mhw_detection_scope': ('pointwise (each grid cell is an independent '
                                '1-D time series; no spatial object tracking)'),
        'software': f'xrMHW v{__version__} (Hobday et al. 2016 definition)',
    })
    for key in ('xrMHW_rate_per_decade', 'xrMHW_t_ref', 'xrMHW_trend',
                'xrMHW_trend_degree', 'xrMHW_period_baselines'):
        if key in clim.attrs:
            ds_out.attrs['mhw_' + key[6:]] = clim.attrs[key]
    return ds_out


# =============================================================================
# 5. ANNUAL METRICS
# =============================================================================

_CATEGORY_NAMES = ('moderate', 'strong', 'severe', 'extreme')


def _annual_block(mask, inten, sev, yi=(), ystart=(), ny=1, cold=False,
                  cum_by_start=True, categories=True):
    """Annual metrics of a block of cells, vectorised over the cells.

    `mask`, `inten` and `sev` are (..., time): the event days, the daily
    intensity and the daily severity. Events are the runs of event days (a
    joined gap is part of its event). Event metrics go to the year in which
    the event starts; days are counted in each calendar year.
    """
    shape = mask.shape[:-1]
    T = mask.shape[-1]
    m = mask.reshape(-1, T).astype(bool)
    n = m.shape[0]
    x = np.where(m, np.nan_to_num(inten.reshape(-1, T).astype(np.float64)), 0.0)
    yi = np.asarray(yi)
    ystart = np.asarray(ystart)

    # Runs of event days: start s (inclusive) and end e (exclusive) per row.
    d = np.diff(m.astype(np.int8), axis=1, prepend=0, append=0)
    rs, s = np.nonzero(d == 1)
    _, e = np.nonzero(d == -1)
    dur = (e - s).astype(np.float64)
    C = np.zeros((n, T + 1))
    np.cumsum(x, axis=1, out=C[:, 1:])
    cum = C[rs, e] - C[rs, s]
    # Peak of each run: the largest anomaly, or the most negative for cold spells.
    sign = -1.0 if cold else 1.0
    flat = np.append(np.where(m, sign * x, -np.inf).ravel(), -np.inf)
    idx = np.empty(2 * len(s), np.int64)
    idx[0::2] = rs * T + s
    idx[1::2] = rs * T + e
    peak = sign * np.maximum.reduceat(flat, idx)[0::2] if len(s) else np.zeros(0)

    key = rs * ny + yi[s]
    size = n * ny

    def per_start_year(w=None):
        return np.bincount(key, weights=w, minlength=size).reshape(n, ny)

    count = per_start_year()
    with np.errstate(invalid='ignore', divide='ignore'):
        safe = np.where(count > 0, count, np.nan)
        duration = per_start_year(dur) / safe
        i_mean = per_start_year(cum / np.maximum(dur, 1)) / safe
        i_max = per_start_year(peak) / safe
        i_cum = per_start_year(cum) / safe
    i_max_max = np.full(size, -np.inf)
    if len(s):
        np.maximum.at(i_max_max, key, sign * peak)
    i_max_max = np.where(np.isfinite(i_max_max), sign * i_max_max, np.nan).reshape(n, ny)

    days = np.add.reduceat(m.astype(np.int32), ystart, axis=1).astype(np.float64)
    if cum_by_start:
        total_icum = per_start_year(cum)
    else:
        total_icum = np.add.reduceat(x, ystart, axis=1)

    out = [count, days, duration, i_mean, i_max, i_max_max, i_cum, total_icum]
    if categories:
        with np.errstate(invalid='ignore'):
            cat = np.floor(sev.reshape(-1, T))
        cat = np.where(m, cat, 0)
        for sel in (cat == 1, cat == 2, cat == 3, cat >= 4):
            out.append(np.add.reduceat(sel.astype(np.int32), ystart,
                                       axis=1).astype(np.float64))
    return tuple(o.reshape(shape + (ny,)) for o in out)


def xrMHW_annual(events, cum_by='start_year', valid=None):
    """
    Annual metrics per grid cell from the daily event fields.

    Takes the output of `xrMHW_events` or `xrMHW_func` and returns, for each
    cell and calendar year, the quantities of marineHeatWaves' `blockAverage`
    (one-year blocks): number of events, event days, mean duration, mean and
    peak intensity, cumulative intensity and, when the severity is present,
    the days in each category. These are the quantities usually mapped,
    trended or compared between baselines (Smith et al. 2025, Box 1).

    Parameters
    ----------
    events : xr.Dataset
        Output of `xrMHW_events` or `xrMHW_func` with ``mhw_intensity`` (or
        ``mcs_intensity``). ``*_duration`` is used, if present, to mark the
        event days; ``*_severity``, if present, gives the category days.
        Event days are the runs of non-missing values, so the fields must not
        be subset in time after detection.
    cum_by : {'start_year', 'calendar_year'}
        Where the total cumulative intensity of the year (``*_total_icum``)
        is counted.

        * ``'start_year'`` (default): the whole cumulative intensity of an
          event goes to the year in which it starts, as stated in
          marineHeatWaves' `blockAverage`. (Its code puts events that cross
          a year into the year in which they end.)
        * ``'calendar_year'``: the daily intensities are summed within each
          calendar year, so an event that crosses a year is split.
    valid : xr.DataArray of bool, optional
        Cells with data, e.g. ``ds['sst'].notnull().any('time')``; the
        others are NaN. By default the cells where ``*_threshold`` or
        ``climatology`` is present when ``events`` has them; otherwise every
        cell counts, and cells without data show zero events.

    Returns
    -------
    xr.Dataset
        On the input grid with ``year`` in place of ``time``. With prefix
        ``mhw`` (or ``mcs``):

        * ``mhw_count`` — events that start in the year;
        * ``mhw_days`` — event days in the calendar year;
        * ``mhw_duration`` — mean duration of the events that start in the
          year (NaN without events);
        * ``mhw_intensity_mean``, ``mhw_intensity_max`` and
          ``mhw_intensity_cumulative`` — means over those events of their
          mean, peak and cumulative intensity;
        * ``mhw_intensity_max_max`` — the largest peak among them (the most
          negative for cold spells);
        * ``mhw_total_icum`` — total cumulative intensity of the year, by
          `cum_by`;
        * ``mhw_moderate_days`` … ``mhw_extreme_days`` — event days in each
          Hobday et al. (2018) category in the calendar year, from the
          severity (floor of S: 1, 2, 3, 4 or more).
    """
    if cum_by not in ('start_year', 'calendar_year'):
        raise ValueError(f"cum_by must be 'start_year' or 'calendar_year', not {cum_by!r}.")
    if not isinstance(events, xr.Dataset):
        raise TypeError("`events` must be the xarray.Dataset returned by "
                        "xrMHW_events or xrMHW_func.")
    found = [p for p in ('mhw', 'mcs') if f'{p}_intensity' in events]
    if len(found) != 1:
        raise ValueError(
            "`events` must contain exactly one of 'mhw_intensity' or "
            "'mcs_intensity' (request it with output_vars=['intensity', ...]).")
    prefix = found[0]
    if 'time' not in events.indexes:
        raise ValueError("`events` must have a 'time' coordinate.")
    time = events.indexes['time']
    if not time.is_monotonic_increasing:
        raise ValueError("The time coordinate must be increasing.")

    inten = events[f'{prefix}_intensity']
    mask_src = events.get(f'{prefix}_duration', inten)
    mask = mask_src.notnull()
    has_sev = f'{prefix}_severity' in events
    sev = events[f'{prefix}_severity'] if has_sev else inten

    years_all = np.asarray(time.year)
    years, yi = np.unique(years_all, return_inverse=True)
    ystart = np.flatnonzero(np.diff(yi, prepend=-1))
    inputs = [mask, inten, sev]
    if inten.chunks is not None:
        inputs = [a.chunk({'time': -1}) for a in inputs]

    names = ['count', 'days', 'duration', 'intensity_mean', 'intensity_max',
             'intensity_max_max', 'intensity_cumulative', 'total_icum']
    if has_sev:
        names += [f'{c}_days' for c in _CATEGORY_NAMES]
    results = xr.apply_ufunc(
        _annual_block, *inputs,
        kwargs=dict(yi=yi, ystart=ystart, ny=len(years), cold=prefix == 'mcs',
                    cum_by_start=cum_by == 'start_year', categories=has_sev),
        input_core_dims=[['time']] * 3,
        output_core_dims=[['year']] * len(names),
        dask='parallelized',
        output_dtypes=[np.float64] * len(names),
        dask_gufunc_kwargs={'output_sizes': {'year': len(years)}},
    )

    if valid is None:
        ref = next((events[v] for v in (f'{prefix}_threshold', 'climatology')
                    if v in events), None)
        if ref is not None:
            valid = ref.notnull().any('time') if 'time' in ref.dims else ref.notnull()

    units = inten.attrs.get('units', 'degC')
    meta = {
        'count': ('Number of events that start in the year', '1'),
        'days': ('Event days in the calendar year', 'days'),
        'duration': ('Mean duration of the events that start in the year', 'days'),
        'intensity_mean': ('Mean of the mean intensities of the events that '
                           'start in the year', units),
        'intensity_max': ('Mean of the peak intensities of the events that '
                          'start in the year', units),
        'intensity_max_max': ('Largest peak intensity among the events that '
                              'start in the year', units),
        'intensity_cumulative': ('Mean of the cumulative intensities of the '
                                 'events that start in the year', f'{units} days'),
        'total_icum': ('Total cumulative intensity of the year (' +
                       ('events counted in their start year'
                        if cum_by == 'start_year' else
                        'daily intensities summed within the calendar year') + ')',
                       f'{units} days'),
    }
    for c in _CATEGORY_NAMES:
        meta[f'{c}_days'] = (f'Event days in category {c} in the calendar year '
                             '(floor of the severity index)', 'days')

    ds_out = xr.Dataset()
    for name, da in zip(names, results):
        if valid is not None:
            da = da.where(valid)
        long_name, u = meta[name]
        ds_out[f'{prefix}_{name}'] = da.assign_attrs(long_name=long_name, units=u)
    ds_out = ds_out.assign_coords(year=np.asarray(years, dtype=np.int32))
    for c in events.coords:
        if c != 'time' and 'time' not in events[c].dims:
            ds_out = ds_out.assign_coords({c: events[c]})
    ds_out.attrs = {k: v for k, v in events.attrs.items()}
    ds_out.attrs.update(mhw_annual_cum_by=cum_by,
                        mhw_annual_note=('Event metrics are counted in the year in '
                                         'which the event starts; days in each '
                                         'calendar year.'))
    return ds_out


# =============================================================================
# 6. TRENDS
# =============================================================================

# Largest pairwise-difference array built at once (elements); the cells of a
# block are processed in groups below this size (~256 MB in float64).
_TREND_MAX_PAIRS = 2 ** 25


def _sen_slope(D, dt):
    """Median of the pairwise slopes, ignoring pairs with a missing value."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN rows
        return np.nanmedian(D / dt, axis=1)


def _mk_variance(X, nv):
    """Mann-Kendall variance of S with the tie correction, per row."""
    m, n = X.shape
    Xs = np.sort(X, axis=1)                      # NaN go last, each its own group
    new = np.concatenate([np.ones((m, 1), bool), Xs[:, 1:] != Xs[:, :-1]], axis=1)
    gid = np.cumsum(new, axis=1) - 1
    cnt = np.bincount((np.arange(m)[:, None] * n + gid).ravel(),
                      minlength=m * n).reshape(m, n).astype(np.float64)
    tsum = (cnt * (cnt - 1) * (2 * cnt + 5)).sum(axis=1)
    return (nv * (nv - 1) * (2 * nv + 5) - tsum) / 18.0


def _hamed_rao_factor(X, t, valid, nv, sen, alpha):
    """Variance correction of Hamed & Rao (1998), as in pymannkendall: the
    autocorrelation of the ranks of the Sen-detrended series, keeping the lags
    outside the confidence band. Missing values are skipped (the valid values
    are taken in order), and the trend is removed at their true times."""
    from scipy.stats import norm, rankdata
    order = np.argsort(~valid, axis=1, kind='stable')
    Xc = np.take_along_axis(X, order, axis=1)
    vc = np.take_along_axis(valid, order, axis=1)
    # Missing values go last as +inf, so they do not change the ranks of the
    # valid ones (rankdata's nan_policy needs SciPy >= 1.10).
    d = np.where(vc, Xc - sen[:, None] * t[order], np.inf)
    ranks = np.where(vc, rankdata(d, axis=1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        y = np.nan_to_num(ranks - np.nanmean(ranks, axis=1, keepdims=True))
    acov0 = (y * y).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        bound = norm.ppf(1 - alpha / 2) / np.sqrt(nv)
        sni = np.zeros(len(X))
        for k in range(1, X.shape[1]):
            acf = (y[:, :-k] * y[:, k:]).sum(axis=1) / acov0
            use = (np.abs(acf) > bound) & (k < nv)
            sni += np.where(use, (nv - k) * (nv - k - 1) * (nv - k - 2) * acf, 0.0)
        factor = 1 + 2 * sni / (nv * (nv - 1) * (nv - 2))
    return np.where(acov0 > 0, factor, 1.0)


def _trend_rows(X, t, method, test, alpha, min_valid):
    """Slope, p-value and number of valid values for each row of X (m, n)."""
    from scipy.stats import norm, t as student_t
    m, n = X.shape
    valid = np.isfinite(X)
    X = np.where(valid, X, np.nan)
    nv = valid.sum(axis=1).astype(np.float64)
    i, j = np.triu_indices(n, 1)
    D = X[:, j] - X[:, i]                        # NaN when either value is missing
    sen = _sen_slope(D, t[j] - t[i]) if (method == 'sen' or test == 'hamed_rao') else None

    if method == 'sen':
        slope = sen
    else:
        tt = np.where(valid, t, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            dt = np.where(valid, t - np.nanmean(tt, axis=1, keepdims=True), 0.0)
            dx = np.where(valid, X - np.nanmean(X, axis=1, keepdims=True), 0.0)
        sxx = (dt * dt).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            slope = (dt * dx).sum(axis=1) / sxx

    with np.errstate(invalid='ignore', divide='ignore'):
        if test is None:
            p = np.full(m, np.nan)
        elif test == 't':
            dof = nv - 2
            resid = dx - slope[:, None] * dt
            se = np.sqrt((resid * resid).sum(axis=1) / dof / sxx)
            p = 2 * student_t.sf(np.abs(slope / se), dof)
            p = np.where(se > 0, p, np.where(slope != 0, 0.0, 1.0))
        else:
            S = np.nansum(np.sign(D), axis=1)
            var = _mk_variance(X, nv)
            if test == 'hamed_rao':
                var = var * _hamed_rao_factor(X, t, valid, nv, sen, alpha)
            # As pymannkendall: z = 0 when S = 0; a negative corrected
            # variance (possible with Hamed-Rao) gives NaN.
            z = np.where(S == 0, 0.0, (S - np.sign(S)) / np.sqrt(var))
            p = 2 * (1 - norm.cdf(np.abs(z)))
    few = nv < max(int(min_valid), 3)
    return (np.where(few, np.nan, slope), np.where(few, np.nan, p), nv)


def _trend_block(X, t=(), method='sen', test='mk', alpha=0.05, min_valid=3):
    """Vectorised trends of a block (..., n), in groups of rows that keep the
    pairwise-difference array below `_TREND_MAX_PAIRS` elements."""
    shape = X.shape[:-1]
    n = X.shape[-1]
    X = X.reshape(-1, n).astype(np.float64)
    t = np.asarray(t, dtype=np.float64)
    rows = max(1, _TREND_MAX_PAIRS // max(n * (n - 1) // 2, 1))
    out = [np.empty(len(X)) for _ in range(3)]
    for a in range(0, len(X), rows):
        for o, r in zip(out, _trend_rows(X[a:a + rows], t, method, test, alpha,
                                          min_valid)):
            o[a:a + rows] = r
    return tuple(o.reshape(shape) for o in out)


def _fdr_bh(p, q):
    """Benjamini-Hochberg: True where the p-value passes at false discovery
    rate `q` over all finite p-values of the field."""
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    ps = np.sort(p[ok])
    m = len(ps)
    passed = ps <= q * np.arange(1, m + 1) / max(m, 1)
    pmax = ps[passed].max() if passed.any() else -1.0
    return ok & (p <= pmax)


def _trend_axis(coord, per):
    """Positions along the trend dimension (1 at the first value, in units of
    the coordinate, or years for dates), the factor that converts a slope per
    position unit into a slope per `per`, and the label of `per`.

    Starting at 1 makes the Hamed-Rao detrending identical to pymannkendall's
    (x - k * slope, k = 1..n) for yearly series, ties included."""
    v = np.asarray(coord.values)
    if np.issubdtype(v.dtype, np.datetime64):
        years = (v - v[0]).astype('timedelta64[s]').astype(np.float64) / (365.2425 * 86400)
        unit = 'year'
    elif np.issubdtype(v.dtype, np.number):
        years, unit = v.astype(np.float64) - float(v[0]), None
    else:
        raise TypeError(f"The coordinate '{coord.name}' must be numeric (e.g. "
                        "years) or datetime.")
    if per == 'decade':
        scale, label = 10.0, 'decade'
    elif per == 'year':
        scale, label = 1.0, 'year'
    elif isinstance(per, (int, float)) and not isinstance(per, bool) and per > 0:
        scale, label = float(per), f'{per:g} {unit or "units"}'
    else:
        raise ValueError(f"per must be 'decade', 'year' or a positive number, not {per!r}.")
    return years + 1.0, scale, label


def xrMHW_trend(data, dim='year', method='sen', test='mk', alpha=0.05,
                fdr=None, per='decade', min_valid=3):
    """
    Trend, its significance and, optionally, field significance, per grid cell.

    Usually applied to the output of `xrMHW_annual`. The computation is
    vectorised over each block of cells (no Python loop per cell) and lazy
    with Dask; the series along `dim` must be whole in each chunk.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Series along `dim`. A Dataset gives the trend of each of its variables
        that has `dim`.
    dim : str
        Dimension of the series, default ``'year'``. Its coordinate must be
        numeric (years) or datetime.
    method : {'sen', 'ols'}
        Slope: Theil-Sen (median of the pairwise slopes; default) or ordinary
        least squares.
    test : {'mk', 'hamed_rao', 't', None}
        * ``'mk'`` (default) — Mann-Kendall with the tie correction, as
          ``pymannkendall.original_test``.
        * ``'hamed_rao'`` — Mann-Kendall with the variance corrected for
          autocorrelation (Hamed & Rao 1998), as
          ``pymannkendall.hamed_rao_modification_test`` with all lags. When
          the corrected variance is negative the p-value is NaN, as there
          (about 1 % of white-noise series of 33 values).
        * ``'t'`` — t-test of the least-squares slope, as
          ``scipy.stats.linregress`` (only with ``method='ols'``).
        * ``None`` — slope only.
    alpha : float
        Significance level of ``significant`` (and of the confidence band of
        the Hamed-Rao correction). Default 0.05.
    fdr : float, optional
        False discovery rate for field significance (Benjamini & Hochberg
        1995; Wilks 2016 recommends 0.10). It ranks the p-values of the whole
        field, so it computes them (with Dask, the result is computed).
    per : {'decade', 'year'} or float
        Unit of the slope: per decade (default), per year, or per that many
        units of the coordinate.
    min_valid : int
        Cells with fewer non-missing values along `dim` are NaN (at least 3).
        Missing values (for example years without events in a mean
        duration) are skipped: the pairs and the variance use the valid
        values of each cell.

    Returns
    -------
    xr.Dataset
        ``slope``, ``p_value``, ``significant`` (p < alpha),
        ``significant_fdr`` (with `fdr`) and ``n_valid``. For a Dataset input
        the names are prefixed with each variable's name (``mhw_days_slope``).
    """
    if method not in ('sen', 'ols'):
        raise ValueError(f"method must be 'sen' or 'ols', not {method!r}.")
    if test not in ('mk', 'hamed_rao', 't', None):
        raise ValueError(f"test must be 'mk', 'hamed_rao', 't' or None, not {test!r}.")
    if test == 't' and method != 'ols':
        raise ValueError("test='t' tests the least-squares slope; use method='ols'.")
    if fdr is not None and test is None:
        raise ValueError("fdr needs a test.")
    if isinstance(data, xr.Dataset):
        names = [v for v in data.data_vars if dim in data[v].dims]
        if not names:
            raise ValueError(f"No variable of the dataset has the dimension '{dim}'.")
        parts = []
        for v in names:
            r = xrMHW_trend(data[v], dim=dim, method=method, test=test, alpha=alpha,
                            fdr=fdr, per=per, min_valid=min_valid)
            parts.append(r.rename({k: f'{v}_{k}' for k in r.data_vars}))
        out = xr.merge(parts, combine_attrs='drop_conflicts')
        return out
    if not isinstance(data, xr.DataArray):
        raise TypeError("`data` must be an xarray.DataArray or Dataset.")
    if dim not in data.dims:
        raise ValueError(f"'{dim}' is not a dimension of the data {data.dims}.")
    if dim not in data.coords:
        data = data.assign_coords({dim: np.arange(data.sizes[dim])})
    t, scale, label = _trend_axis(data[dim], per)
    if data.chunks is not None:
        data = data.chunk({dim: -1})

    slope, p, nv = xr.apply_ufunc(
        _trend_block, data,
        kwargs=dict(t=t, method=method, test=test, alpha=alpha, min_valid=min_valid),
        input_core_dims=[[dim]],
        output_core_dims=[[], [], []],
        dask='parallelized',
        output_dtypes=[np.float64] * 3,
    )
    units = data.attrs.get('units', '')
    method_txt = {'sen': 'Theil-Sen', 'ols': 'least squares'}[method]
    test_txt = {'mk': 'Mann-Kendall', 'hamed_rao': 'Mann-Kendall, Hamed-Rao correction',
                't': 't-test of the least-squares slope', None: 'none'}[test]
    out = xr.Dataset()
    out['slope'] = (slope * scale).assign_attrs(
        long_name=f'{method_txt} slope per {label}',
        units=f'{units} per {label}'.strip())
    if test is not None:
        out['p_value'] = p.assign_attrs(long_name=f'p-value ({test_txt}, two-sided)',
                                        units='1')
        out['significant'] = (p < alpha).where(p.notnull()).assign_attrs(
            long_name=f'p-value below {alpha}', units='1')
        if fdr is not None:
            pv = np.asarray(p.values)
            out['significant_fdr'] = xr.DataArray(
                np.where(np.isfinite(pv), _fdr_bh(pv, fdr), np.nan),
                dims=p.dims, coords=p.coords).assign_attrs(
                long_name=f'Significant at false discovery rate {fdr} '
                          '(Benjamini-Hochberg over the field)', units='1')
    out['n_valid'] = nv.assign_attrs(long_name=f'Non-missing values along {dim}',
                                     units='1')
    out.attrs.update(trend_method=method, trend_test=str(test), trend_alpha=alpha,
                     trend_per=label, trend_min_valid=int(min_valid))
    if fdr is not None:
        out.attrs['trend_fdr'] = fdr
    return out
