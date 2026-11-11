"""
Tests of the baselines of Smith et al. (2025, Prog. Oceanogr. 231, 103404):
fixed, detrended, adaptation-adjusted and periodically updated.

Identities (E):
* adaptation with rate 0 is the fixed baseline; with a rate, the threshold
  rises by that rate per decade from the centre of the reference period;
* the detrended baseline applied to the raw field gives the events of the
  fixed baseline applied to the field from xrMHW_detrend;
* each period of the periodic baseline uses the fixed climatology of the
  reference period before it.

Properties of Smith et al., Box 1 (H-J), on synthetic daily series built
like theirs (seasonal cycle, warming trend, autocorrelated variability) and
measured, as they do, as the share of days above the threshold:
* H fixed: about 10 % in the reference period, more afterwards;
* I detrended with the right trend shape: about 10 % in every decade;
* J detrended with a linear trend on a nonlinear warming: spurious changes,
  which a polynomial of the right degree removes.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import xrMHW

xr = pytest.importorskip("xarray")

REF = (1961, 1990)


def synthetic(trend="none", amp=2.0, n_lat=4, n_lon=5, seed=1):
    """Daily 1961-2020 series on a small grid: seasonal cycle, AR(1)
    variability (phi = 0.9, sd 0.8 degC) independent in each cell, and a
    warming of `amp` degC over the record, linear or quadratic."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1961-01-01", "2020-12-31", freq="D")
    n = len(dates)
    s = np.linspace(0.0, 1.0, n)
    warming = {"none": 0 * s, "linear": amp * s, "quadratic": amp * s ** 2}[trend]
    seasonal = 12.0 + 4.0 * np.sin(2 * np.pi * (dates.dayofyear.values - 80) / 365.25)
    phi, sd = 0.9, 0.8
    eps = rng.normal(0.0, sd * np.sqrt(1 - phi ** 2), (n, n_lat, n_lon))
    noise = np.empty_like(eps)
    noise[0] = rng.normal(0.0, sd, (n_lat, n_lon))
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + eps[i]
    temp = seasonal[:, None, None] + warming[:, None, None] + noise
    da = xr.DataArray(temp, dims=("time", "lat", "lon"),
                      coords={"time": dates, "lat": np.arange(n_lat, dtype=float),
                              "lon": np.arange(n_lon, dtype=float)}, name="sst")
    return da.to_dataset()


def exceed_by_decade(ds, clim):
    """Share of days above the threshold, by decade, over all cells."""
    thr = xrMHW.xrMHW_events(ds, "sst", clim, output_vars=["threshold"])["mhw_threshold"]
    above = (ds["sst"] > thr)
    return above.groupby((ds["time"].dt.year // 10) * 10).mean(...).to_series()


# ---------------------------------------------------------------- E: identities
def test_adaptation_rate_zero_is_fixed():
    ds = synthetic("linear", n_lat=1, n_lon=2)
    fixed = xrMHW.xrMHW_events(ds, "sst", xrMHW.xrMHW_clim(ds, "sst", REF))
    adapt = xrMHW.xrMHW_events(ds, "sst", xrMHW.xrMHW_clim(
        ds, "sst", REF, baseline="adaptation", rate=0.0))
    for v in fixed.data_vars:
        assert np.array_equal(fixed[v].values, adapt[v].values, equal_nan=True), v


def test_adaptation_threshold_rises_at_rate():
    ds = synthetic("linear", n_lat=1, n_lon=1)
    fixed = xrMHW.xrMHW_events(ds, "sst", xrMHW.xrMHW_clim(ds, "sst", REF),
                               output_vars=["threshold", "climatology"])
    clim = xrMHW.xrMHW_clim(ds, "sst", REF, baseline="adaptation", rate=0.3)
    # centre of 1961-01-01 .. 1990-12-31 (5478 days after the first day)
    assert clim.attrs["xrMHW_t_ref"] == "1976-01-01"
    adapt = xrMHW.xrMHW_events(ds, "sst", clim, output_vars=["threshold", "climatology"])
    t = np.array([d.toordinal() for d in ds.indexes["time"]], float)
    expected = 0.3 * (t - clim.attrs["xrMHW_offset_t0"]) / 3652.5
    diff = (adapt["mhw_threshold"] - fixed["mhw_threshold"]).values.ravel()
    assert np.allclose(diff, expected, atol=1e-5)
    # only the threshold moves: the climatology stays fixed
    assert np.array_equal(adapt["climatology"].values, fixed["climatology"].values)


@pytest.mark.parametrize("trend,degree", [("linear", None), ("polynomial", 3)])
def test_detrended_baseline_equals_detrended_field(trend, degree):
    ds = synthetic("quadratic", amp=3.0, n_lat=2, n_lon=2)
    clim = xrMHW.xrMHW_clim(ds, "sst", REF, baseline="detrended", trend=trend,
                            degree=degree)
    on_raw = xrMHW.xrMHW_events(ds, "sst", clim)

    det = xrMHW.xrMHW_detrend(ds, "sst", REF, trend=trend, degree=degree)
    assert np.allclose(det["trend_coef"].values, clim["offset_coef"].values)
    on_detrended = xrMHW.xrMHW_events(det, "sst", xrMHW.xrMHW_clim(det, "sst", REF))

    assert np.array_equal(on_raw["mhw_duration"].notnull().values,
                          on_detrended["mhw_duration"].notnull().values)
    for v in on_raw.data_vars:
        assert np.allclose(on_raw[v].values, on_detrended[v].values,
                           atol=1e-5, equal_nan=True), v


def test_detrended_trend_has_zero_mean_in_reference_period():
    ds = synthetic("linear", n_lat=1, n_lon=1)
    det = xrMHW.xrMHW_detrend(ds, "sst", REF)
    offset = (ds["sst"] - det["sst"]).sel(time=slice("1961", "1990"))
    assert abs(float(offset.mean())) < 1e-10


def test_periodic_uses_the_previous_reference_period():
    ds = synthetic("linear", n_lat=1, n_lon=2)
    with pytest.warns(UserWarning, match="first complete one"):
        clim = xrMHW.xrMHW_clim(ds, "sst", baseline="periodic")
    assert clim["period"].values.tolist() == [1961, 1971, 1981, 1991, 2001, 2011]
    per_date = xrMHW.xrMHW_events(ds, "sst", clim, output_vars=["threshold"])

    for years, ref in ((slice("2011", "2020"), (1981, 2010)),
                       (slice("1991", "2000"), (1961, 1990)),
                       (slice("1961", "1970"), (1961, 1990))):   # first complete window
        fixed = xrMHW.xrMHW_events(ds, "sst", xrMHW.xrMHW_clim(ds, "sst", ref),
                                   output_vars=["threshold"])
        assert np.array_equal(per_date["mhw_threshold"].sel(time=years).values,
                              fixed["mhw_threshold"].sel(time=years).values), ref


def test_xrMHW_func_with_a_baseline():
    ds = synthetic("linear", n_lat=1, n_lon=2)
    one = xrMHW.xrMHW_func(ds, "sst", REF, baseline="adaptation", rate=0.2)
    two = xrMHW.xrMHW_events(ds, "sst", xrMHW.xrMHW_clim(
        ds, "sst", REF, baseline="adaptation", rate=0.2),
        output_vars=list(one.data_vars))
    assert set(one.data_vars) == {"mhw_intensity", "mhw_severity", "mhw_duration",
                                  "mhw_category", "mhw_max_intensity",
                                  "mhw_cum_intensity", "climatology", "mhw_threshold"}
    for v in one.data_vars:
        assert np.array_equal(one[v].values, two[v].values, equal_nan=True), v
    assert one.attrs["mhw_baseline"] == "adaptation"


# ---------------------------------------------------------------- H-J: Smith et al., Box 1
def test_fixed_baseline_saturates_with_warming():
    ds = synthetic("linear", amp=2.0)
    frac = exceed_by_decade(ds, xrMHW.xrMHW_clim(ds, "sst", REF))
    in_ref = float(frac.loc[[1960, 1970, 1980]].mean())
    assert 0.085 < in_ref < 0.115, frac
    assert frac.loc[2010] > 0.25, frac


def test_detrended_with_the_right_trend_keeps_ten_percent():
    ds = synthetic("linear", amp=2.0)
    frac = exceed_by_decade(ds, xrMHW.xrMHW_clim(ds, "sst", REF, baseline="detrended"))
    assert frac.between(0.07, 0.13).all(), frac


def test_detrended_with_the_wrong_trend_shape_is_spurious():
    ds = synthetic("quadratic", amp=3.0)
    linear = exceed_by_decade(ds, xrMHW.xrMHW_clim(ds, "sst", REF, baseline="detrended"))
    poly = exceed_by_decade(ds, xrMHW.xrMHW_clim(ds, "sst", REF, baseline="detrended",
                                                 trend="polynomial", degree=2))
    assert (linear - 0.10).abs().max() > 0.04, linear
    assert poly.between(0.07, 0.13).all(), poly


@pytest.mark.parametrize("baseline,opts", [("detrended", {}), ("adaptation", {"rate": 0.2}),
                                           ("periodic", {})])
def test_baselines_with_dask(baseline, opts):
    pytest.importorskip("dask")
    ds = synthetic("linear", n_lat=2, n_lon=2)
    period = None if baseline == "periodic" else REF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # periodic: first complete window
        eager = xrMHW.xrMHW_func(ds, "sst", period, baseline=baseline, **opts)
        lazy = xrMHW.xrMHW_func(ds.chunk({"time": -1, "lat": 1, "lon": 2}), "sst",
                                period, baseline=baseline, **opts)
    assert lazy["mhw_duration"].chunks is not None
    lazy = lazy.compute()
    for v in eager.data_vars:
        assert np.array_equal(eager[v].values, lazy[v].values, equal_nan=True), v


# ---------------------------------------------------------------- errors
def test_baseline_errors():
    ds = synthetic(n_lat=1, n_lon=1)
    with pytest.raises(ValueError, match="baseline must be one of"):
        xrMHW.xrMHW_clim(ds, "sst", REF, baseline="moving")
    with pytest.raises(ValueError, match="needs rate"):
        xrMHW.xrMHW_clim(ds, "sst", REF, baseline="adaptation")
    with pytest.raises(ValueError, match="degree >= 1"):
        xrMHW.xrMHW_clim(ds, "sst", REF, baseline="detrended", trend="polynomial")
    with pytest.raises(ValueError, match="degree 1"):
        xrMHW.xrMHW_clim(ds, "sst", REF, baseline="detrended", degree=2)
    with pytest.raises(ValueError, match="needs clim_period"):
        xrMHW.xrMHW_clim(ds, "sst", baseline="fixed")
    with pytest.raises(ValueError, match="shorter than ref_length"):
        xrMHW.xrMHW_clim(ds.sel(time=slice("1990", "2010")), "sst", baseline="periodic")
    with pytest.raises(TypeError, match="unexpected keyword"):
        xrMHW.xrMHW_clim(ds, "sst", REF, rate=0.1)
