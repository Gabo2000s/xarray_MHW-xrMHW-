"""
Tests of xrMHW_annual (annual metrics per cell).

* F: with ``cum_by='start_year'`` every quantity equals marineHeatWaves'
  ``blockAverage`` year by year, except where ``blockAverage``'s code departs
  from its own documentation (an event that crosses a year has its
  cumulative intensity counted in the year it ends).
* The vectorised block computation equals a plain loop over the cells.
* ``'calendar_year'`` sums the daily intensities within each year; both
  conventions have the same total over the record.
* Dask gives the same result as in memory; cells without data are NaN;
  cold spells; wrong inputs.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi

import xrMHW

xr = pytest.importorskip("xarray")

CLIM_PERIOD = (1991, 2020)
NAMES = ["count", "days", "duration", "intensity_mean", "intensity_max",
         "intensity_max_max", "intensity_cumulative", "total_icum",
         "moderate_days", "strong_days", "severe_days", "extreme_days"]


@pytest.fixture(scope="module")
def series():
    """30 years of daily SST with persistent (AR(1)) anomalies, so there are
    many events, plus injected ones, one crossing 2009/2010."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    doy = dates.dayofyear.values
    noise = np.empty(len(dates))
    noise[0] = 0.0
    eps = rng.normal(0.0, 0.8 * np.sqrt(1 - 0.9 ** 2), len(dates))
    for i in range(1, len(dates)):
        noise[i] = 0.9 * noise[i - 1] + eps[i]
    temp = 15.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.25) + noise
    for start, length, amp in (("2010-07-01", 12, 4.0), ("2005-09-10", 6, 2.2),
                               ("2018-02-15", 8, 3.0), ("2009-12-25", 20, 4.5)):
        i = np.where(dates == np.datetime64(start))[0][0]
        temp[i:i + length] += amp
    return dates, temp


def _one_cell(dates, temp):
    return xr.DataArray(temp[:, None, None], dims=("time", "lat", "lon"),
                        coords={"time": dates, "lat": [45.0], "lon": [-50.0]},
                        name="sst").to_dataset()


@pytest.fixture(scope="module")
def grid(series):
    """(time, lat=2, lon=3) with different noise per cell and one land cell."""
    dates, temp = series
    rng = np.random.default_rng(7)
    cube = np.empty((len(dates), 2, 3))
    for j in range(2):
        for k in range(3):
            cube[:, j, k] = temp + rng.normal(0.0, 0.3, len(dates))
    cube[:, 1, 2] = np.nan
    return xr.DataArray(cube, dims=("time", "lat", "lon"),
                        coords={"time": dates, "lat": [44.0, 45.0],
                                "lon": [-50.0, -49.0, -48.0]},
                        name="sst").to_dataset()


# ---------------------------------------------------------------- F
@pytest.fixture(scope="module")
def against_oliver(series):
    mhw_mod = pytest.importorskip("marineHeatWaves")
    dates, temp = series
    t = np.array([d.toordinal() for d in dates])
    ev, clim = mhw_mod.detect(t, temp.copy(), climatologyPeriod=list(CLIM_PERIOD))
    blk = mhw_mod.blockAverage(t, ev, clim=clim, temp=temp)
    events = xrMHW.xrMHW_func(_one_cell(dates, temp), "sst", CLIM_PERIOD)
    ann = xrMHW.xrMHW_annual(events).isel(lat=0, lon=0)
    return ev, blk, ann


def test_years_match_blockAverage(against_oliver):
    _, blk, ann = against_oliver
    assert np.array_equal(ann["year"].values, blk["years_start"].astype(int))


@pytest.mark.parametrize("ours, theirs, atol", [
    ("count", "count", 0), ("days", "total_days", 0),
    ("duration", "duration", 0),
    ("intensity_mean", "intensity_mean", 1e-4),
    ("intensity_max", "intensity_max", 1e-4),
    ("intensity_max_max", "intensity_max_max", 1e-4),
    ("intensity_cumulative", "intensity_cumulative", 1e-3),
    ("moderate_days", "moderate_days", 0), ("strong_days", "strong_days", 0),
    ("severe_days", "severe_days", 0), ("extreme_days", "extreme_days", 0),
])
def test_equals_blockAverage(against_oliver, ours, theirs, atol):
    _, blk, ann = against_oliver
    a = ann[f"mhw_{ours}"].values
    b = np.asarray(blk[theirs], dtype=float)
    if ours == "intensity_max_max":      # blockAverage leaves 0 where count is 0
        b = np.where(blk["count"] > 0, b, np.nan)
    assert np.allclose(a, b, atol=atol, rtol=0, equal_nan=True), (
        f"{ours}: max diff {np.nanmax(np.abs(a - b))}")


def test_total_icum_by_start_year(against_oliver):
    """Equal to blockAverage except for events that cross a year, which
    blockAverage's code puts in the year they end (its comment says the
    start year); xrMHW follows the documented convention."""
    ev, blk, ann = against_oliver
    years = ann["year"].values
    ours = ann["mhw_total_icum"].values
    expected = np.zeros(len(years))
    crossing = []
    for i in range(ev["n_events"]):
        y0, y1 = ev["date_start"][i].year, ev["date_end"][i].year
        expected[years == y0] += ev["intensity_cumulative"][i]
        if y0 != y1:
            crossing.append((y0, y1, ev["intensity_cumulative"][i]))
    assert np.allclose(ours, expected, atol=1e-3)
    assert crossing, "the fixture must contain an event that crosses a year"
    theirs = np.asarray(blk["total_icum"], dtype=float)
    shifted = expected.copy()
    for y0, y1, icum in crossing:
        shifted[years == y0] -= icum
        shifted[years == y1] += icum
    assert np.allclose(theirs, shifted, atol=1e-6)
    same = ~np.isin(years, [c[0] for c in crossing] + [c[1] for c in crossing])
    assert np.allclose(ours[same], theirs[same], atol=1e-3)


# ---------------------------------------------------------------- conventions
def test_calendar_year_sums_daily_intensity(grid):
    events = xrMHW.xrMHW_func(grid, "sst", CLIM_PERIOD)
    cal = xrMHW.xrMHW_annual(events, cum_by="calendar_year")
    start = xrMHW.xrMHW_annual(events)
    daily = events["mhw_intensity"].fillna(0).groupby("time.year").sum()
    land = grid["sst"].isnull().all("time")
    assert np.allclose(cal["mhw_total_icum"].values,
                       daily.where(~land).transpose(*cal["mhw_total_icum"].dims).values,
                       atol=1e-3, equal_nan=True)
    tot_cal = cal["mhw_total_icum"].sum("year")
    tot_start = start["mhw_total_icum"].sum("year")
    assert np.allclose(tot_cal.values, tot_start.values, atol=1e-2, equal_nan=True)
    # only the cumulative total depends on the convention
    for n in NAMES:
        if n != "total_icum":
            assert np.array_equal(cal[f"mhw_{n}"].values, start[f"mhw_{n}"].values,
                                  equal_nan=True)


# ---------------------------------------------------------------- vectorised vs loop
def _loop_cell(dates, duration, inten, sev):
    """Plain per-event loop for one cell: the reference for the block code."""
    years = np.unique(dates.year)
    out = {n: np.zeros(len(years)) for n in NAMES}
    peaks = [[] for _ in years]
    labels, n = ndi.label(~np.isnan(duration))
    for k in range(1, n + 1):
        idx = np.where(labels == k)[0]
        j = np.searchsorted(years, dates.year[idx[0]])
        x = inten[idx]
        out["count"][j] += 1
        out["duration"][j] += len(idx)
        out["intensity_mean"][j] += x.mean()
        out["intensity_max"][j] += x.max()
        out["intensity_cumulative"][j] += x.sum()
        out["total_icum"][j] += x.sum()
        peaks[j].append(x.max())
    for j, y in enumerate(years):
        sel = (dates.year == y) & ~np.isnan(duration)
        out["days"][j] = sel.sum()
        cat = np.floor(sev[sel])
        out["moderate_days"][j] = (cat == 1).sum()
        out["strong_days"][j] = (cat == 2).sum()
        out["severe_days"][j] = (cat == 3).sum()
        out["extreme_days"][j] = (cat >= 4).sum()
    c = np.where(out["count"] > 0, out["count"], np.nan)
    for n_ in ("duration", "intensity_mean", "intensity_max", "intensity_cumulative"):
        out[n_] = out[n_] / c
    out["intensity_max_max"] = np.array([max(p) if p else np.nan for p in peaks])
    return out


def test_block_equals_loop(grid):
    events = xrMHW.xrMHW_func(grid, "sst", CLIM_PERIOD)
    ann = xrMHW.xrMHW_annual(events)
    dates = events.indexes["time"]
    for j in range(2):
        for k in range(3):
            if j == 1 and k == 2:
                continue
            cell = events.isel(lat=j, lon=k)
            ref = _loop_cell(dates, cell["mhw_duration"].values.astype(float),
                             cell["mhw_intensity"].values.astype(float),
                             cell["mhw_severity"].values.astype(float))
            for n in NAMES:
                got = ann[f"mhw_{n}"].isel(lat=j, lon=k).values
                assert np.allclose(got, ref[n], atol=1e-4, equal_nan=True), (n, j, k)


# ---------------------------------------------------------------- dask, masks, cold spells
def test_dask_equals_memory(grid):
    pytest.importorskip("dask")
    events = xrMHW.xrMHW_func(grid, "sst", CLIM_PERIOD)
    mem = xrMHW.xrMHW_annual(events)
    lazy = xrMHW.xrMHW_annual(events.chunk({"time": 1000, "lat": 1, "lon": 2}))
    assert lazy["mhw_count"].chunks is not None
    for n in NAMES:
        assert np.allclose(lazy[f"mhw_{n}"].values, mem[f"mhw_{n}"].values,
                           equal_nan=True), n


def test_cells_without_data(grid):
    events = xrMHW.xrMHW_func(grid, "sst", CLIM_PERIOD)
    ann = xrMHW.xrMHW_annual(events)               # threshold present: land is NaN
    assert ann["mhw_count"].isel(lat=1, lon=2).isnull().all()
    assert ann["mhw_count"].isel(lat=0, lon=0).notnull().all()
    only = events[["mhw_intensity"]]               # no threshold: land shows zeros
    assert (xrMHW.xrMHW_annual(only)["mhw_count"].isel(lat=1, lon=2) == 0).all()
    valid = grid["sst"].notnull().any("time")
    assert xrMHW.xrMHW_annual(only, valid=valid)["mhw_count"].isel(
        lat=1, lon=2).isnull().all()
    assert "mhw_moderate_days" not in xrMHW.xrMHW_annual(only)


def test_cold_spells(grid):
    events = xrMHW.xrMHW_func(grid, "sst", CLIM_PERIOD, cold_spells=True)
    ann = xrMHW.xrMHW_annual(events)
    assert "mcs_count" in ann and "mhw_count" not in ann
    peak = ann["mcs_intensity_max"].values
    assert np.nanmax(peak) < 0
    assert np.all((ann["mcs_intensity_max_max"] <= ann["mcs_intensity_max"] + 1e-9)
                  | ann["mcs_intensity_max"].isnull())
    assert (ann["mcs_total_icum"].fillna(0) <= 0).all()


def test_wrong_inputs(grid):
    events = xrMHW.xrMHW_func(grid, "sst", CLIM_PERIOD)
    with pytest.raises(ValueError, match="cum_by"):
        xrMHW.xrMHW_annual(events, cum_by="end_year")
    with pytest.raises(ValueError, match="mhw_intensity"):
        xrMHW.xrMHW_annual(events[["mhw_duration"]])
    with pytest.raises(TypeError):
        xrMHW.xrMHW_annual(events["mhw_intensity"])
