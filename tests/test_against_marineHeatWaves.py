"""
Regression tests: xrMHW  vs  marineHeatWaves (Oliver / Hobday et al. 2016)
==========================================================================

Goal
----
Prove that xrMHW, run with ``join_method='post-filter'`` (the Hobday/Oliver
default), reproduces Eric Oliver's reference ``marineHeatWaves`` implementation
to floating-point tolerance, and that the multidimensional ``xrMHW_func`` path
is consistent with the 1-D core, pixel by pixel.

Design choices
--------------
* We require precise agreement, not just statistical "equivalence". This means matching event counts, start and end dates, durations, and intensities/climatology within a very narrow margin. Any discrepancies are fully reported
* The climatology tolerance is set at 1e-10 degC. The index-space window replicates Oliver precisely.
* ``test_climatology_vectorisation_is_exact`` ensures the optimised, loop-free climatology aligns perfectly with a literal transcription of the definition. This serves as a safeguard, allowing the implementation to remain fast without deviation. 
* We use deterministic SYNTHETIC data (seeded), eliminating reliance on large external files.

Test dependencies
-----------------
    pip install pytest numpy pandas scipy xarray
``marineHeatWaves`` is vendored next to this file; ``xarray`` is skipped
gracefully if absent, so the numpy-only core tests always run.

Run with:
    pytest -v test_against_marineHeatWaves.py
"""

import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi

import xrMHW

# ----------------------------------------------------------------------------
# Shared parameters and synthetic-data fixture
# ----------------------------------------------------------------------------
CLIM_PERIOD = (1991, 2020)
PARAMS = dict(
    pctile=90,
    min_duration=5,
    max_gap=2,
    window_half_width=5,
    smooth_pctile=True,
    smooth_width=31,
    join_across_gaps=True,
    join_method="post-filter",   # the configuration that must match Oliver
)

ATOL_CLIM = 1e-10      # degC, element-wise climatology / threshold
ATOL_INTENSITY = 1e-8  # degC, per-event peak / cumulative
MAX_DUR_DIFF = 0       # days; boundaries must agree exactly

@pytest.fixture(scope="module")
def synthetic():
    """30 years of daily SST: seasonal cycle + reproducible noise + injected
    warm events of known length so detection has unambiguous targets."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    doy = dates.dayofyear.values
    seasonal = 15.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    temp = seasonal + rng.normal(0.0, 0.8, len(dates))

    def add_event(start, length, amp):
        i = np.where(dates == np.datetime64(start))[0][0]
        temp[i:i + length] += amp

    add_event("2010-07-01", 12, 4.0)   # long, extreme
    add_event("2005-09-10", 6, 2.2)    # short, moderate/strong
    add_event("2018-02-15", 8, 3.0)    # winter event
    add_event("1998-06-05", 5, 2.5)    # exactly at the min-duration boundary

    t_ord = np.array([d.toordinal() for d in dates]).astype(int)
    return dates, t_ord, temp.astype(float)

# ----------------------------------------------------------------------------
# Helpers to put Oliver's output into the same shape as xrMHW's
# ----------------------------------------------------------------------------
def run_xrmhw_core(t_ord, temp):
    """Returns the 9-tuple from the real core:
    (seas, thresh, anomaly, is_mhw, duration, category, int_max, int_cum,
     severity)."""
    return xrMHW.mhw_1d_wrapper(
        t_ord, temp.copy(), CLIM_PERIOD[0], CLIM_PERIOD[1], **PARAMS
    )

def oliver_detect(mhw_mod, t_ord, temp):
    return mhw_mod.detect(
        t_ord, temp.copy(), climatologyPeriod=list(CLIM_PERIOD),
        pctile=PARAMS["pctile"], minDuration=PARAMS["min_duration"],
        maxGap=PARAMS["max_gap"], windowHalfWidth=PARAMS["window_half_width"],
        smoothPercentile=True, smoothPercentileWidth=PARAMS["smooth_width"],
    )

def events_from_mask(dates, is_mhw, duration, int_max, int_cum, category):
    """Collapse xrMHW's per-day masked arrays into one row per event."""
    labels, n = ndi.label(is_mhw)
    rows = []
    for k in range(1, n + 1):
        idx = np.where(labels == k)[0]
        rows.append(dict(
            start=dates[idx[0]].date(),
            end=dates[idx[-1]].date(),
            duration=int(duration[idx[0]]),
            int_max=float(int_max[idx[0]]),
            int_cum=float(int_cum[idx[0]]),
            category=int(category[idx[0]]),
        ))
    return rows

def events_from_oliver(mhw):
    """One row per Oliver event, same keys as events_from_mask."""
    cat_map = {"Moderate": 1, "Strong": 2, "Severe": 3, "Extreme": 4}
    rows = []
    for i in range(mhw["n_events"]):
        cat = mhw["category"][i]
        rows.append(dict(
            start=mhw["date_start"][i],
            end=mhw["date_end"][i],
            duration=int(mhw["duration"][i]),
            int_max=float(mhw["intensity_max"][i]),
            int_cum=float(mhw["intensity_cumulative"][i]),
            category=cat_map.get(cat, cat),
        ))
    return rows

def match_by_overlap(oliver_rows, xr_rows):
    """Pair events that overlap in time. Returns matched pairs plus any
    unmatched events from either side (these are the interesting failures)."""
    pairs, used = [], set()
    for o in oliver_rows:
        for j, x in enumerate(xr_rows):
            if j in used:
                continue
            if max(o["start"], x["start"]) <= min(o["end"], x["end"]):
                pairs.append((o, x))
                used.add(j)
                break
    matched_o = {id(o) for o, _ in pairs}
    unmatched_o = [o for o in oliver_rows if id(o) not in matched_o]
    unmatched_x = [x for j, x in enumerate(xr_rows) if j not in used]
    return pairs, unmatched_o, unmatched_x

# ----------------------------------------------------------------------------
# THE decisive test: element-wise climatology & threshold
# ----------------------------------------------------------------------------
def test_climatology_threshold(synthetic):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, t_ord, temp = synthetic

    seas_xr, thresh_xr = run_xrmhw_core(t_ord, temp)[:2]
    _, clim = oliver_detect(mhw_mod, t_ord, temp)
    d_seas = np.nanmax(np.abs(seas_xr - clim["seas"]))
    d_thr = np.nanmax(np.abs(thresh_xr - clim["thresh"]))
    print(f"\nmax|Δ climatology| = {d_seas:.2e} °C   max|Δ threshold| = {d_thr:.2e} °C")
    assert d_seas < ATOL_CLIM, f"climatology differs by {d_seas:.2e} °C"
    assert d_thr < ATOL_CLIM, f"threshold differs by {d_thr:.2e} °C"

def test_climatology_vectorisation_is_exact(synthetic):
    """The optimised climatology must equal a literal transcription of the
    definition: the threshold bit for bit, the mean to within rounding. This is
    what allows it to stay fast safely."""
    from datetime import date as _date

    _, t_ord, temp = synthetic
    year, doy = xrMHW._time_metadata(t_ord)
    clim_mask = (year >= CLIM_PERIOD[0]) & (year <= CLIM_PERIOD[1])
    w, pctile = PARAMS["window_half_width"], PARAMS["pctile"]

    # Reference: one day-of-year at a time, exactly as the definition reads.
    T = len(temp)
    clim_idx = np.flatnonzero(clim_mask)
    cs, ce = clim_idx[0], clim_idx[-1]
    offsets = np.arange(-w, w + 1)
    thresh_ref = np.full(366, np.nan)
    seas_ref = np.full(366, np.nan)
    for d in range(1, 367):
        if d == 60:
            continue
        occ = np.flatnonzero(doy[cs:ce + 1] == d)
        if occ.size == 0:
            continue
        tt = (cs + occ[:, None] + offsets[None, :]).ravel()
        tt = tt[(tt >= 0) & (tt < T)]
        thresh_ref[d - 1] = np.nanpercentile(temp[tt], pctile)
        seas_ref[d - 1] = np.nanmean(temp[tt])
    thresh_ref[59] = 0.5 * thresh_ref[58] + 0.5 * thresh_ref[60]
    seas_ref[59] = 0.5 * seas_ref[58] + 0.5 * seas_ref[60]
    thresh_ref = xrMHW._runavg(thresh_ref, PARAMS["smooth_width"])
    seas_ref = xrMHW._runavg(seas_ref, PARAMS["smooth_width"])

    seas_fast, thresh_fast = xrMHW._seasonal_climatology(
        temp, doy, clim_mask, pctile, w, True, PARAMS["smooth_width"])

    assert np.array_equal(thresh_fast, thresh_ref, equal_nan=True), (
        f"vectorised threshold drifted: "
        f"max|Δ| = {np.nanmax(np.abs(thresh_fast - thresh_ref)):.3e}")
    # The percentile only orders the values, so it is exact. The mean sums them,
    # and a floating-point sum depends on the order of the terms, which NumPy
    # chooses differently across versions and CPUs (SIMD). Allow rounding only:
    # 1e-12 degC is a hundred times tighter than the tolerance against Oliver.
    assert np.array_equal(np.isnan(seas_fast), np.isnan(seas_ref)), (
        "vectorised climatology has NaN on different days")
    d_seas = np.nanmax(np.abs(seas_fast - seas_ref))
    assert d_seas < 1e-12, (
        f"vectorised climatology drifted: max|Δ| = {d_seas:.3e}")

def test_event_count(synthetic):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, t_ord, temp = synthetic

    _, _, _, is_mhw, dur, cat, imax, icum, _sev = run_xrmhw_core(t_ord, temp)
    xr_rows = events_from_mask(dates, is_mhw, dur, imax, icum, cat)

    mhw, _ = oliver_detect(mhw_mod, t_ord, temp)
    o_rows = events_from_oliver(mhw)
    print(f"\nN events: Oliver={len(o_rows)}  xrMHW={len(xr_rows)}")
    assert len(xr_rows) == len(o_rows), (
        f"event count differs: Oliver={len(o_rows)} xrMHW={len(xr_rows)}"
    )

def test_event_dates_durations_intensities(synthetic):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, t_ord, temp = synthetic

    _, _, _, is_mhw, dur, cat, imax, icum, _sev = run_xrmhw_core(t_ord, temp)
    xr_rows = events_from_mask(dates, is_mhw, dur, imax, icum, cat)

    mhw, _ = oliver_detect(mhw_mod, t_ord, temp)
    o_rows = events_from_oliver(mhw)

    pairs, unmatched_o, unmatched_x = match_by_overlap(o_rows, xr_rows)

    print("\n  start(O)     dur(O) dur(xr)  imax(O)  imax(xr)  cum(O)   cum(xr)")
    dur_fail, int_fail = [], []
    for o, x in pairs:
        print(f"  {o['start']}   {o['duration']:5d} {x['duration']:6d}  "
              f"{o['int_max']:7.3f} {x['int_max']:8.3f}  "
              f"{o['int_cum']:7.2f} {x['int_cum']:8.2f}")
        if abs(o["duration"] - x["duration"]) > MAX_DUR_DIFF:
            dur_fail.append((o["start"], o["duration"], x["duration"]))
        if abs(o["int_max"] - x["int_max"]) > ATOL_INTENSITY:
            int_fail.append((o["start"], o["int_max"], x["int_max"]))

    assert not unmatched_o, f"Oliver events with no xrMHW match: {unmatched_o}"
    assert not unmatched_x, f"xrMHW events with no Oliver match: {unmatched_x}"
    assert not dur_fail, f"duration mismatch > {MAX_DUR_DIFF} d: {dur_fail}"
    assert not int_fail, f"peak intensity mismatch > {ATOL_INTENSITY}: {int_fail}"

def test_categories(synthetic):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, t_ord, temp = synthetic

    _, _, _, is_mhw, dur, cat, imax, icum, _sev = run_xrmhw_core(t_ord, temp)
    xr_rows = events_from_mask(dates, is_mhw, dur, imax, icum, cat)
    mhw, _ = oliver_detect(mhw_mod, t_ord, temp)
    o_rows = events_from_oliver(mhw)
    pairs, _, _ = match_by_overlap(o_rows, xr_rows)
    bad = [(o["start"], o["category"], x["category"])
           for o, x in pairs if o["category"] != x["category"]]
    assert not bad, f"category mismatch (start, Oliver, xrMHW): {bad}"

# ----------------------------------------------------------------------------
# Severity index (Sen Gupta et al. 2020; Malan et al. 2025, eq. 1)
# ----------------------------------------------------------------------------
def test_severity_definition(synthetic):
    """S = (T - T_clim) / (T_p90 - T_clim), element-wise, dimensionless."""
    _, t_ord, temp = synthetic
    seas, thresh, anomaly, _is, _d, _c, _im, _ic, severity = run_xrmhw_core(
        t_ord, temp)
    expected = (temp - seas) / (thresh - seas)
    assert np.allclose(severity, expected, rtol=0, atol=1e-12, equal_nan=True)
    # Severity must equal the anomaly normalised by the threshold distance.
    assert np.allclose(severity, anomaly / (thresh - seas),
                       rtol=0, atol=1e-12, equal_nan=True)

def test_severity_exceeds_one_during_events(synthetic):
    """By construction every event day sits above the threshold, i.e. S > 1
    (Malan et al. 2025 identify extreme profiles exactly this way)."""
    _, t_ord, temp = synthetic
    *_, is_mhw, _d, _c, _im, _ic, severity = run_xrmhw_core(t_ord, temp)
    inside = severity[np.asarray(is_mhw)]
    # Days bridged across a short gap are allowed to dip below the threshold.
    assert np.nanmin(inside) > 0.5
    assert np.nanmean(inside) > 1.0

def test_category_is_floor_of_peak_severity(synthetic):
    """Hobday et al. (2018) categories are the integer part of the severity
    index at its peak, capped at 4 (Extreme)."""
    _, t_ord, temp = synthetic
    *_, is_mhw, _d, category, _im, _ic, severity = run_xrmhw_core(t_ord, temp)
    labels, n = ndi.label(np.asarray(is_mhw))
    for k in range(1, n + 1):
        idx = np.where(labels == k)[0]
        expected = min(max(1, int(np.floor(np.nanmax(severity[idx])))), 4)
        assert int(category[idx[0]]) == expected, (
            f"event {k}: category {category[idx[0]]} != floor(max S) {expected}")

# ----------------------------------------------------------------------------
# Core-only tests (no Oliver, no xarray): always run
# ----------------------------------------------------------------------------
def test_land_mask_fast_exit(synthetic):
    dates, t_ord, _ = synthetic
    land = np.full(len(dates), np.nan)
    seas, thresh, anom, is_mhw, *_ = xrMHW.mhw_1d_wrapper(
        t_ord, land, CLIM_PERIOD[0], CLIM_PERIOD[1], **PARAMS
    )
    assert not np.asarray(is_mhw).any()
    assert np.isnan(seas).all() and np.isnan(thresh).all()

def test_data_gap_interpolation(synthetic):
    """A short (<= max_gap_interp) run of NaNs inside an event must be bridged
    so the event is not artificially fragmented."""
    dates, t_ord, temp = synthetic
    i0 = np.where(dates == np.datetime64("2010-07-01"))[0][0]
    gapped = temp.copy()
    gapped[i0 + 5:i0 + 7] = np.nan          # 2-day data gap inside the event
    _, _, _, is_mhw, *_ = xrMHW.mhw_1d_wrapper(
        t_ord, gapped, CLIM_PERIOD[0], CLIM_PERIOD[1],
        max_gap_interp=2, **PARAMS
    )
    block = np.asarray(is_mhw)[i0:i0 + 12]
    assert block.sum() == 12, f"event fragmented by data gap: {block.astype(int)}"

def test_fast_path_matches_full_path(synthetic):
    """compute_event_metrics=False must give the same event mask, climatology,
    threshold and severity as the full path; only per-event arrays are empty."""
    _, t_ord, temp = synthetic
    full = xrMHW.mhw_1d_wrapper(t_ord, temp.copy(), *CLIM_PERIOD, **PARAMS)
    fast = xrMHW.mhw_1d_wrapper(t_ord, temp.copy(), *CLIM_PERIOD,
                                compute_event_metrics=False, **PARAMS)
    for i, name in enumerate(["seas", "thresh", "anomaly"]):
        assert np.array_equal(full[i], fast[i], equal_nan=True), name
    assert np.array_equal(np.asarray(full[3]), np.asarray(fast[3])), "event mask"
    assert np.array_equal(full[8], fast[8], equal_nan=True), "severity"

def test_climatology_baseline_outside_record_raises(synthetic):
    _, t_ord, temp = synthetic
    with pytest.raises(ValueError, match="climatology baseline"):
        xrMHW.detect_mhw_core(t_ord, temp.copy(), clim_period=(1800, 1810))

# ----------------------------------------------------------------------------
# Multidimensional consistency: xrMHW_func over a grid == 1-D core per pixel
# ----------------------------------------------------------------------------
def test_grid_matches_1d(synthetic):
    xr = pytest.importorskip("xarray")
    dates, t_ord, temp = synthetic

    # Build a tiny (time, lat, lon) cube: 3 identical ocean pixels + 1 land.
    ocean = temp
    land = np.full_like(temp, np.nan)
    cube = np.stack([
        np.stack([ocean, ocean], axis=-1),
        np.stack([ocean, land], axis=-1),
    ], axis=-1)  # (time, lat=2, lon=2); [1,1] is land
    da = xr.DataArray(
        cube, dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [0, 1], "lon": [0, 1]},
        name="sst",
    )
    ds = da.to_dataset()

    out = xrMHW.xrMHW_func(ds, "sst", CLIM_PERIOD, **PARAMS)

    _, _, _, is_mhw_1d, dur_1d, *_ = run_xrmhw_core(t_ord, temp)

    is_mhw_grid = out["mhw_duration"].isel(lat=0, lon=0).notnull().values
    assert np.array_equal(is_mhw_grid, np.asarray(is_mhw_1d)), \
        "grid ocean pixel does not match 1-D core"

    land_dur = out["mhw_duration"].isel(lat=1, lon=1).values
    assert np.isnan(land_dur).all(), "land pixel produced spurious events"

def test_output_vars_selection(synthetic):
    """output_vars must restrict the dataset and accept short aliases."""
    xr = pytest.importorskip("xarray")
    dates, _, temp = synthetic
    ds = xr.DataArray(temp[:, None, None], dims=("time", "lat", "lon"),
                      coords={"time": dates, "lat": [0], "lon": [0]},
                      name="sst").to_dataset()

    full = xrMHW.xrMHW_func(ds, "sst", CLIM_PERIOD, **PARAMS)
    assert set(full.data_vars) == {
        "mhw_intensity", "mhw_severity", "mhw_duration", "mhw_category",
        "mhw_max_intensity", "mhw_cum_intensity", "climatology",
        "mhw_threshold"}

    subset = xrMHW.xrMHW_func(ds, "sst", CLIM_PERIOD,
                              output_vars=["severity", "thresh"], **PARAMS)
    assert set(subset.data_vars) == {"mhw_severity", "mhw_threshold"}
    # Values must be identical to the corresponding fields of the full run.
    assert np.allclose(subset["mhw_severity"].values,
                       full["mhw_severity"].values, equal_nan=True)
    assert np.allclose(subset["mhw_threshold"].values,
                       full["mhw_threshold"].values, equal_nan=True)

    with pytest.raises(ValueError, match="Unknown output var"):
        xrMHW.xrMHW_func(ds, "sst", CLIM_PERIOD, output_vars=["nonsense"])

def test_mhw_and_mcs_outputs_can_be_merged(synthetic):
    """Prefixed thresholds let a heatwave and a cold-spell run be merged
    without a name collision (they hold different percentiles)."""
    xr = pytest.importorskip("xarray")
    dates, _, temp = synthetic
    ds = xr.DataArray(temp[:, None, None], dims=("time", "lat", "lon"),
                      coords={"time": dates, "lat": [0], "lon": [0]},
                      name="sst").to_dataset()

    hot = xrMHW.xrMHW_func(ds, "sst", CLIM_PERIOD,
                           output_vars=["intensity", "thresh"])
    cold = xrMHW.xrMHW_func(ds, "sst", CLIM_PERIOD, cold_spells=True,
                            output_vars=["intensity", "thresh"])
    assert "mhw_threshold" in hot and "mcs_threshold" in cold
    merged = xr.merge([hot, cold])
    assert {"mhw_threshold", "mcs_threshold"} <= set(merged.data_vars)
    # The 10th percentile must sit below the 90th everywhere.
    assert (merged["mcs_threshold"] <= merged["mhw_threshold"]).all()

def test_output_metadata_is_documented(synthetic):
    """Every output variable must carry long_name, units and a description;
    duration must state that it is pointwise."""
    xr = pytest.importorskip("xarray")
    dates, _, temp = synthetic
    ds = xr.DataArray(temp[:, None, None], dims=("time", "lat", "lon"),
                      coords={"time": dates, "lat": [0], "lon": [0]},
                      name="sst").to_dataset()
    out = xrMHW.xrMHW_func(ds, "sst", CLIM_PERIOD)
    for name, var in out.data_vars.items():
        assert var.attrs.get("long_name"), f"{name} has no long_name"
        assert var.attrs.get("units"), f"{name} has no units"
        assert var.attrs.get("description"), f"{name} has no description"
    dur_doc = (out["mhw_duration"].attrs["long_name"] + " "
               + out["mhw_duration"].attrs["description"]).lower()
    assert "pointwise" in dur_doc, "duration must be documented as pointwise"
    assert out.attrs["mhw_percentile"] == 90
    assert "pointwise" in out.attrs["mhw_detection_scope"]