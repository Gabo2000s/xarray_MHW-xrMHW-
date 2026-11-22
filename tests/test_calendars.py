"""Calendar handling: only the standard (Gregorian) calendar is supported.

Before this check, a 'noleap' time axis returned no events without any error,
and a cftime axis in the standard calendar was read with wrong years.
"""
import numpy as np
import pytest
import xarray as xr

import xrMHW

pytest.importorskip("cftime")


def _series(calendar, use_cftime=True):
    end = "2009-12-30" if calendar == "360_day" else "2009-12-31"
    if use_cftime:
        try:
            time = xr.date_range("1990-01-01", end, freq="D",
                                 calendar=calendar, use_cftime=True)
        except (AttributeError, TypeError):  # older xarray
            time = xr.cftime_range("1990-01-01", end, freq="D",
                                   calendar=calendar)
    else:
        time = xr.date_range("1990-01-01", "2009-12-31", freq="D") \
            if hasattr(xr, "date_range") else \
            __import__("pandas").date_range("1990-01-01", "2009-12-31", freq="D")
    rng = np.random.default_rng(0)
    n = len(time)
    temp = (10 + 3 * np.sin(2 * np.pi * np.arange(n) / 365.25)
            + np.convolve(rng.normal(0, 1, n + 9), np.ones(10) / 3, "valid"))
    return xr.Dataset({"sst": ("time", temp)}, coords={"time": time})


@pytest.mark.parametrize("calendar", ["noleap", "360_day", "julian"])
@pytest.mark.parametrize("func", ["func", "clim"])
def test_non_standard_calendar_raises(calendar, func):
    ds = _series(calendar)
    with pytest.raises(ValueError, match="supports only the standard"):
        if func == "func":
            xrMHW.xrMHW_func(ds, "sst", (1990, 2009))
        else:
            xrMHW.xrMHW_clim(ds, "sst", (1990, 2009))


def test_cftime_standard_calendar_equals_numpy_datetimes():
    expected = xrMHW.xrMHW_func(_series(None, use_cftime=False), "sst", (1990, 2009))
    got = xrMHW.xrMHW_func(_series("standard"), "sst", (1990, 2009))
    assert np.isfinite(expected["mhw_intensity"]).sum() > 0
    for name in expected.data_vars:
        np.testing.assert_array_equal(got[name].values, expected[name].values)


def test_suggested_conversion_runs():
    ds = _series("noleap").convert_calendar("standard", align_on="year",
                                            missing=np.nan)
    out = xrMHW.xrMHW_func(ds, "sst", (1990, 2009))
    assert np.isfinite(out["mhw_intensity"]).sum() > 0
