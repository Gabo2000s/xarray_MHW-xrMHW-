"""xrMHW_func with a time axis split into several Dask chunks.

The climatology and the event detection use each cell's complete record, so the
whole time axis must sit in one chunk. A split time axis is rechunked with a
UserWarning instead of failing inside xarray.apply_ufunc.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("dask")

import xrMHW  # noqa: E402

CLIM = (2000, 2005)
MESSAGE = "whole time axis in one chunk"


def make_dataset():
    """3 x 2 cells over six years: a seasonal sinusoid plus noise, with two
    injected warm events."""
    dates = pd.date_range("2000-01-01", "2005-12-31", freq="D")
    doy = dates.dayofyear.values
    rng = np.random.default_rng(7)
    seasonal = 12.0 + 4.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    temp = seasonal[:, None, None] + rng.normal(0.0, 0.4, (dates.size, 3, 2))
    i0 = np.where(dates == np.datetime64("2002-08-01"))[0][0]
    i1 = np.where(dates == np.datetime64("2004-02-10"))[0][0]
    temp[i0:i0 + 12] += 4.0          # every cell
    temp[i1:i1 + 8, 1, :] += 5.0     # one row of cells
    da = xr.DataArray(temp, coords={"time": dates, "latitude": [40.0, 41.0, 42.0],
                                    "longitude": [-50.0, -49.0]},
                      dims=["time", "latitude", "longitude"])
    return xr.Dataset({"thetao": da})


def _our_warnings(record):
    return [w for w in record
            if issubclass(w.category, UserWarning) and MESSAGE in str(w.message)]


def test_split_time_axis_is_rechunked_with_a_warning():
    ds = make_dataset()
    whole = xrMHW.xrMHW_func(ds.chunk({"time": -1}), "thetao", CLIM).compute()
    with pytest.warns(UserWarning, match=MESSAGE):
        split = xrMHW.xrMHW_func(ds.chunk({"time": 365}), "thetao", CLIM)
    xr.testing.assert_identical(split.compute(), whole)
    # the injected events are found, so the comparison is not between empty fields
    assert int(whole["mhw_duration"].notnull().sum()) > 0


def test_split_time_axis_matches_the_in_memory_result():
    ds = make_dataset()
    eager = xrMHW.xrMHW_func(ds, "thetao", CLIM)
    with pytest.warns(UserWarning, match=MESSAGE):
        split = xrMHW.xrMHW_func(ds.chunk({"time": 400, "latitude": 1}), "thetao",
                                 CLIM).compute()
    for name in eager.data_vars:
        np.testing.assert_allclose(eager[name].values, split[name].values, equal_nan=True)


def test_single_time_chunk_does_not_warn():
    ds = make_dataset()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        xrMHW.xrMHW_func(ds.chunk({"time": -1, "latitude": 1}), "thetao", CLIM).compute()
    assert not _our_warnings(record)


def test_numpy_input_does_not_warn():
    ds = make_dataset()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        xrMHW.xrMHW_func(ds, "thetao", CLIM)
    assert not _our_warnings(record)
