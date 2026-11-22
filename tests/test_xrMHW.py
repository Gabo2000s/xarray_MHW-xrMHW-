"""Unit tests for the xarray/Dask interface of xrMHW.

These describe the behavior a user experiences with ``xrMHW_func``: detecting on a grid, 
handling gaps, using land masks, identifying cold spells, calculating the severity index, 
and choosing outputs. The consistency with Hobday et al.'s (2016) reference implementation 
is checked separately in ``test_against_marineHeatWaves.py``.

Note on the synthetic data
--------------------------
The series include a seasonal cycle and added noise, so the baseline shows real variability.
 A flat, constant baseline is not useful: the seasonal mean and the 90th percentile match, 
 and whether a day is an exceedance is determined by rounding noise at the 1e-15 degC level 
 instead of the data. Such a series doesn't test the definition, so it is only used in the test 
 that explicitly documents this degenerate case.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xrMHW

CLIM = (1990, 1992)

def make_series(years=4, seed=0, sigma=0.5):
    """Daily SST with a seasonal cycle and reproducible noise."""
    dates = pd.date_range("1990-01-01", periods=365 * years, freq="D")
    doy = dates.dayofyear.values
    rng = np.random.default_rng(seed)
    temp = (15.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
            + rng.normal(0.0, sigma, len(dates)))
    return dates, temp.astype(float)

def as_dataset(dates, temp):
    """Wrap a 1-D series as a (time, lat, lon) dataset with one ocean pixel."""
    da = xr.DataArray(temp[:, np.newaxis, np.newaxis],
                      coords=[dates, [0], [0]], dims=["time", "lat", "lon"])
    return xr.Dataset({"thetao": da})

def event_days(res, prefix="mhw"):
    return ~np.isnan(res[f"{prefix}_intensity"].values[0, 0, :])

def test_mhw_detection_basic():
    """A 6-day warm spike injected after the baseline must be detected, and
    only it."""
    dates, temp = make_series()
    i0 = np.where(dates == np.datetime64("1993-07-01"))[0][0]
    temp[i0:i0 + 6] += 6.0

    res = xrMHW.xrMHW_func(as_dataset(dates, temp), 'thetao', CLIM,
                           min_duration=5, max_gap=2, join_across_gaps=True)

    for name in ('mhw_intensity', 'mhw_severity', 'mhw_duration',
                 'mhw_category', 'climatology', 'mhw_threshold'):
        assert name in res

    is_mhw = event_days(res)
    assert is_mhw[i0:i0 + 6].all(), "the injected event was not detected"
    assert is_mhw.sum() == 6, (
        f"{is_mhw.sum()} event days detected, expected only the injected 6")
    assert res['mhw_duration'].values[0, 0, i0] == 6

def test_gap_joining_methods():
    """pre-filter joins two sub-threshold-length spikes; post-filter (the
    Hobday/Oliver default) discards them."""
    dates, temp = make_series()
    i0 = np.where(dates == np.datetime64("1993-07-01"))[0][0]
    temp[i0 - 3:i0] -= 3.0        # cold fence, so nothing merges from the left
    temp[i0:i0 + 4] += 6.0        # spike 1 (4 days)
    temp[i0 + 4:i0 + 6] -= 3.0    # 2-day gap, clearly below threshold
    temp[i0 + 6:i0 + 10] += 6.0   # spike 2 (4 days)
    temp[i0 + 10:i0 + 13] -= 3.0  # cold fence on the right

    ds = as_dataset(dates, temp)

    res_post = xrMHW.xrMHW_func(ds, 'thetao', CLIM, min_duration=5, max_gap=2,
                                join_across_gaps=True, join_method='post-filter')
    assert not event_days(res_post).any(), (
        "post-filter must discard both 4-day spikes")

    res_pre = xrMHW.xrMHW_func(ds, 'thetao', CLIM, min_duration=5, max_gap=2,
                               join_across_gaps=True, join_method='pre-filter')
    is_pre = event_days(res_pre)
    assert is_pre[i0:i0 + 10].all(), "pre-filter must bridge the 2-day gap"
    assert res_pre['mhw_duration'].values[0, 0, i0] == 10

def test_nan_land_mask():
    """Land cells (all NaN) short-circuit and return NaN everywhere."""
    dates, _ = make_series(years=2)
    res = xrMHW.xrMHW_func(as_dataset(dates, np.full(len(dates), np.nan)),
                           'thetao', (1990, 1991))

    assert np.isnan(res['mhw_intensity'].values).all()
    assert np.isnan(res['mhw_severity'].values).all()
    assert np.isnan(res['climatology'].values).all()
    assert np.isnan(res['mhw_threshold'].values).all()

def test_cold_spells():
    """Marine cold spells are detected with cold_spells=True, and severity
    stays positive because numerator and denominator are both negative."""
    dates, temp = make_series()
    i0 = np.where(dates == np.datetime64("1993-07-01"))[0][0]
    temp[i0 - 3:i0] += 3.0        # warm fence, so nothing merges from the left
    temp[i0:i0 + 6] -= 6.0
    temp[i0 + 6:i0 + 9] += 3.0    # warm fence on the right

    res = xrMHW.xrMHW_func(as_dataset(dates, temp), 'thetao', CLIM,
                           cold_spells=True, min_duration=5)

    assert {'mcs_intensity', 'mcs_duration', 'mcs_threshold',
            'mcs_severity'} <= set(res.data_vars)
    assert res.attrs['mhw_type'] == 'Cold Spell'
    assert res.attrs['mhw_percentile'] == 10

    is_mcs = event_days(res, 'mcs')
    assert is_mcs[i0:i0 + 6].all()
    assert is_mcs.sum() == 6

    assert np.all(res['mcs_intensity'].values[0, 0, i0:i0 + 6] < 0)
    assert np.all(res['mcs_severity'].values[0, 0, i0:i0 + 6] > 1.0)

def test_severity_is_dimensionless_and_matches_category():
    """severity = anomaly / (threshold - climatology); the Hobday et al. (2018)
    category is its integer part at the peak, capped at 4."""
    dates, temp = make_series()
    i0 = np.where(dates == np.datetime64("1993-07-01"))[0][0]
    temp[i0:i0 + 6] += 6.0

    res = xrMHW.xrMHW_func(as_dataset(dates, temp), 'thetao', CLIM,
                           min_duration=5)
    sev = res['mhw_severity'].values[0, 0, :]
    inten = res['mhw_intensity'].values[0, 0, :]
    clim = res['climatology'].values[0, 0, :]
    thresh = res['mhw_threshold'].values[0, 0, :]

    np.testing.assert_allclose(sev, inten / (thresh - clim),
                               rtol=1e-5, atol=1e-5, equal_nan=True)
    assert res['mhw_severity'].attrs['units'] == '1'

    peak_sev = np.nanmax(sev)
    assert peak_sev > 1.0, "an event day must sit above the threshold"
    assert res['mhw_category'].values[0, 0, i0] == min(
        max(1, int(np.floor(peak_sev))), 4)

def test_output_vars_skips_event_loop():
    """Selecting only continuous fields returns exactly those variables."""
    dates, temp = make_series()
    res = xrMHW.xrMHW_func(as_dataset(dates, temp), 'thetao', CLIM,
                           output_vars=['severity', 'climatology'])
    assert set(res.data_vars) == {'mhw_severity', 'climatology'}

def test_missing_variable_raises():
    dates, temp = make_series(years=2)
    with pytest.raises(KeyError):
        xrMHW.xrMHW_func(as_dataset(dates, temp), 'does_not_exist', (1990, 1991))

def test_dask_chunked_input_matches_eager():
    """The Dask path must give the same answer as the in-memory path."""
    dates, temp = make_series()
    i0 = np.where(dates == np.datetime64("1993-07-01"))[0][0]
    temp[i0:i0 + 6] += 6.0
    ds = as_dataset(dates, temp)

    eager = xrMHW.xrMHW_func(ds, 'thetao', CLIM)
    lazy = xrMHW.xrMHW_func(ds.chunk({'time': -1, 'lat': 1, 'lon': 1}),
                            'thetao', CLIM).compute()

    for name in eager.data_vars:
        np.testing.assert_allclose(eager[name].values, lazy[name].values,
                                   equal_nan=True)

def test_constant_series_is_not_all_event():
    """Degenerate case: with a perfectly flat baseline the seasonal mean and
    the percentile threshold coincide, so no day may be reported as an event.
    Without the rounding guard in the detection step, the smoothing leaves the
    threshold ~1e-15 degC below the data and the entire record is flagged."""
    dates = pd.date_range("1990-01-01", periods=365 * 2, freq="D")
    temp = np.full(len(dates), 10.0)
    temp[400:406] = 15.0  # the only genuine exceedance

    res = xrMHW.xrMHW_func(as_dataset(dates, temp), 'thetao', (1990, 1990),
                           min_duration=5, max_gap=2, join_across_gaps=True)
    is_mhw = event_days(res)
    assert is_mhw[400:406].all(), "the real spike must still be detected"
    assert is_mhw.sum() == 6, (
        f"{is_mhw.sum()} event days on a flat baseline: rounding noise is "
        f"being read as exceedance")
