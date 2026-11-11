"""
Tests of the modular API: xrMHW_clim, xrMHW_events and xrMHW_doy.

* A: climatology and events in two steps give exactly what xrMHW_func gives
  in one (in memory and with Dask, for heatwaves and cold spells).
* B: the stored climatology equals marineHeatWaves.
* C: a climatology from another series equals marineHeatWaves with
  ``alternateClimatology``.
* D: a threshold given per date equals the same threshold given per day of year.
* G: wrong inputs fail with a clear message.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi

import xrMHW

xr = pytest.importorskip("xarray")

CLIM_PERIOD = (1991, 2020)
EVENT_VARS = ["mhw_intensity", "mhw_severity", "mhw_duration", "mhw_category",
              "mhw_max_intensity", "mhw_cum_intensity"]


@pytest.fixture(scope="module")
def series():
    """30 years of daily SST with a seasonal cycle, noise and injected events."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    doy = dates.dayofyear.values
    seasonal = 15.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    temp = seasonal + rng.normal(0.0, 0.8, len(dates))
    for start, length, amp in (("2010-07-01", 12, 4.0), ("2005-09-10", 6, 2.2),
                               ("2018-02-15", 8, 3.0), ("1998-06-05", 5, -3.0)):
        i = np.where(dates == np.datetime64(start))[0][0]
        temp[i:i + length] += amp
    return dates, temp


@pytest.fixture(scope="module")
def grid(series):
    """(time, lat=2, lon=3) cube: different noise per cell, one land cell, and
    gaps of 2 days (filled) and 10 days (left as NaN)."""
    dates, temp = series
    rng = np.random.default_rng(7)
    cube = np.empty((len(dates), 2, 3))
    for j in range(2):
        for k in range(3):
            cube[:, j, k] = temp + rng.normal(0.0, 0.3, len(dates))
    cube[:, 1, 2] = np.nan                      # land
    cube[400:402, 0, 1] = np.nan                # short gap: interpolated
    cube[5000:5010, 1, 0] = np.nan              # long gap: stays NaN
    da = xr.DataArray(cube.astype(np.float32), dims=("time", "lat", "lon"),
                      coords={"time": dates, "lat": [44.0, 45.0],
                              "lon": [-50.0, -49.0, -48.0]}, name="sst")
    return da.to_dataset()


def _assert_same(a, b, names):
    for name in names:
        assert np.array_equal(a[name].values, b[name].values, equal_nan=True), (
            f"{name} differs between the two-step and the one-step detection")


# ---------------------------------------------------------------- A
@pytest.mark.parametrize("cold", [False, True])
def test_two_steps_equal_xrMHW_func_in_memory(grid, cold):
    prefix = "mcs" if cold else "mhw"
    names = [n.replace("mhw", prefix) for n in EVENT_VARS] + [
        "climatology", f"{prefix}_threshold"]
    one = xrMHW.xrMHW_func(grid, "sst", CLIM_PERIOD, cold_spells=cold)
    clim = xrMHW.xrMHW_clim(grid, "sst", CLIM_PERIOD, cold_spells=cold)
    two = xrMHW.xrMHW_events(grid, "sst", clim, output_vars=names)
    _assert_same(one, two, names)
    assert int(one[f"{prefix}_duration"].notnull().sum()) > 0


def test_two_steps_equal_xrMHW_func_with_dask(grid):
    pytest.importorskip("dask")
    lazy = grid.chunk({"time": -1, "lat": 1, "lon": 2})
    one = xrMHW.xrMHW_func(lazy, "sst", CLIM_PERIOD).compute()
    clim = xrMHW.xrMHW_clim(lazy, "sst", CLIM_PERIOD)
    assert clim["climatology"].chunks is not None, "xrMHW_clim should stay lazy"
    two = xrMHW.xrMHW_events(lazy, "sst", clim).compute()
    _assert_same(one, two, EVENT_VARS)


def test_climatology_with_other_chunks(grid):
    pytest.importorskip("dask")
    clim = xrMHW.xrMHW_clim(grid.chunk({"time": -1, "lat": 2, "lon": 1}), "sst",
                            CLIM_PERIOD)
    lazy = grid.chunk({"time": -1, "lat": 1, "lon": 3})
    out = xrMHW.xrMHW_events(lazy, "sst", clim).compute()
    ref = xrMHW.xrMHW_func(grid, "sst", CLIM_PERIOD)
    _assert_same(ref, out, EVENT_VARS)


def test_clim_output_layout(grid):
    clim = xrMHW.xrMHW_clim(grid, "sst", CLIM_PERIOD)
    assert set(clim.data_vars) == {"climatology", "mhw_threshold"}
    assert clim["mhw_threshold"].dims == ("lat", "lon", "doy")
    assert clim.sizes["doy"] == 366 and int(clim["doy"][0]) == 1
    assert clim["climatology"].dtype == np.float64
    assert clim.attrs["xrMHW_baseline"] == "fixed"
    assert clim.attrs["xrMHW_pctile"] == 90
    assert bool(clim["mhw_threshold"].isel(lat=1, lon=2).isnull().all()), \
        "land cell must have a NaN climatology"
    out = xrMHW.xrMHW_events(grid, "sst", clim)
    assert set(out.data_vars) == set(EVENT_VARS)
    assert out.attrs["mhw_baseline"] == "fixed"
    assert out.attrs["mhw_detection_period"] == str(CLIM_PERIOD)


# ---------------------------------------------------------------- B and C
def _events(dates, out):
    """One row per event of a single-cell output."""
    d = out["mhw_duration"].values.ravel()
    labels, n = ndi.label(np.isfinite(d))
    rows = []
    for k in range(1, n + 1):
        idx = np.flatnonzero(labels == k)
        rows.append((dates[idx[0]].date(), dates[idx[-1]].date(), int(d[idx[0]]),
                     float(out["mhw_max_intensity"].values.ravel()[idx[0]]),
                     float(out["mhw_cum_intensity"].values.ravel()[idx[0]]),
                     int(out["mhw_category"].values.ravel()[idx[0]])))
    return rows


def _oliver_rows(mhw):
    cat = {"Moderate": 1, "Strong": 2, "Severe": 3, "Extreme": 4}
    return [(mhw["date_start"][i], mhw["date_end"][i], int(mhw["duration"][i]),
             float(mhw["intensity_max"][i]), float(mhw["intensity_cumulative"][i]),
             cat[mhw["category"][i]]) for i in range(mhw["n_events"])]


def _one_cell(dates, temp):
    return xr.DataArray(temp[:, None, None], dims=("time", "lat", "lon"),
                        coords={"time": dates, "lat": [0.0], "lon": [0.0]},
                        name="sst").to_dataset()


def test_clim_equals_marineHeatWaves(series):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, temp = series
    t = np.array([d.toordinal() for d in dates])
    _, ref = mhw_mod.detect(t, temp.copy(), climatologyPeriod=list(CLIM_PERIOD))
    clim = xrMHW.xrMHW_clim(_one_cell(dates, temp), "sst", CLIM_PERIOD)
    on_dates = clim.sel(doy=xrMHW.xrMHW_doy(dates))
    seas = on_dates["climatology"].values.ravel()
    thr = on_dates["mhw_threshold"].values.ravel()
    assert np.nanmax(np.abs(seas - ref["seas"])) < 1e-10
    assert np.nanmax(np.abs(thr - ref["thresh"])) < 1e-10


# The vendored reference converts arrays to scalars in this code path, which
# NumPy >= 1.25 flags as deprecated; the warning is about the reference, not xrMHW.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_alternate_climatology_equals_marineHeatWaves(series):
    """Climatology from series B, events in series A: marineHeatWaves'
    ``alternateClimatology``."""
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, temp_a = series
    rng = np.random.default_rng(3)
    temp_b = temp_a - 0.4 + rng.normal(0.0, 0.5, len(dates))
    t = np.array([d.toordinal() for d in dates])

    ref, _ = mhw_mod.detect(t, temp_a.copy(), climatologyPeriod=list(CLIM_PERIOD),
                            alternateClimatology=[t, temp_b.copy()])
    clim_b = xrMHW.xrMHW_clim(_one_cell(dates, temp_b), "sst", CLIM_PERIOD)
    out = xrMHW.xrMHW_events(_one_cell(dates, temp_a), "sst", clim_b)

    ours, theirs = _events(dates, out), _oliver_rows(ref)
    assert len(ours) == len(theirs) > 0
    for a, b in zip(ours, theirs):
        assert a[:3] == b[:3], f"event dates or duration differ: {a} vs {b}"
        assert abs(a[3] - b[3]) < 1e-5 and abs(a[4] - b[4]) < 1e-4, (a, b)
        assert a[5] == b[5], f"category differs: {a} vs {b}"


# ---------------------------------------------------------------- D
def test_threshold_per_date_equals_per_doy(grid):
    clim = xrMHW.xrMHW_clim(grid, "sst", CLIM_PERIOD)
    per_date = clim.sel(doy=xrMHW.xrMHW_doy(grid["time"])).drop_vars("doy")
    assert "time" in per_date["mhw_threshold"].dims
    a = xrMHW.xrMHW_events(grid, "sst", clim)
    b = xrMHW.xrMHW_events(grid, "sst", per_date)
    _assert_same(a, b, EVENT_VARS)


def test_hand_made_threshold(grid):
    """A constant (absolute) threshold built by hand is accepted and gives the
    events of the 1-D core with that threshold."""
    clim = xrMHW.xrMHW_clim(grid, "sst", CLIM_PERIOD)
    fixed = clim.copy()
    fixed["mhw_threshold"] = xr.full_like(clim["mhw_threshold"], 18.0)
    out = xrMHW.xrMHW_events(grid, "sst", fixed)

    temp = grid["sst"].isel(lat=0, lon=0).values.astype(float)
    t = np.array([d.toordinal() for d in grid.indexes["time"]])
    _, doy = xrMHW._time_metadata(t)
    seas = clim["climatology"].isel(lat=0, lon=0).values[doy - 1]
    ref = xrMHW._events_1d(temp, seas, np.full(len(temp), 18.0))
    assert np.array_equal(out["mhw_duration"].isel(lat=0, lon=0).notnull().values,
                          ref[3])


def test_doy_grid():
    days = pd.to_datetime(["2019-02-28", "2019-03-01", "2020-02-29",
                           "2020-03-01", "2019-12-31", "2020-12-31"])
    assert xrMHW.xrMHW_doy(days).values.tolist() == [59, 61, 60, 61, 366, 366]


# ---------------------------------------------------------------- G
def test_clear_errors(grid):
    clim = xrMHW.xrMHW_clim(grid, "sst", CLIM_PERIOD)
    with pytest.raises(TypeError, match="xarray.Dataset"):
        xrMHW.xrMHW_events(grid, "sst", clim["mhw_threshold"])
    with pytest.raises(ValueError, match="exactly one of"):
        xrMHW.xrMHW_events(grid, "sst", clim[["climatology"]])
    with pytest.raises(ValueError, match="Coordinate 'lat' differs"):
        xrMHW.xrMHW_events(grid, "sst", clim.assign_coords(lat=[0.0, 1.0]))
    with pytest.raises(ValueError, match="size"):
        xrMHW.xrMHW_events(grid, "sst", clim.isel(lon=[0, 1]))
    with pytest.raises(ValueError, match="length 366"):
        xrMHW.xrMHW_events(grid, "sst", clim.isel(doy=slice(0, 365)))
    with pytest.raises(ValueError, match="same 'time'"):
        per_date = clim.sel(doy=xrMHW.xrMHW_doy(grid["time"])).drop_vars("doy")
        xrMHW.xrMHW_events(grid, "sst", per_date.isel(time=slice(1, None)))
    with pytest.raises(TypeError, match="unexpected keyword"):
        xrMHW.xrMHW_events(grid, "sst", clim, cold_spells=True)
    with pytest.raises(TypeError, match="unexpected keyword"):
        xrMHW.xrMHW_clim(grid, "sst", CLIM_PERIOD, min_duration=5)
