"""
Tests of xrMHW_trend (vectorised trends per cell).

* Sen slope and Mann-Kendall equal ``pymannkendall.original_test`` cell by
  cell, with ties and with missing values.
* The Hamed-Rao correction equals ``pymannkendall.hamed_rao_modification_test``
  without missing values, and a direct per-cell version with them.
* Least squares and its t-test equal ``scipy.stats.linregress``.
* Field significance equals Benjamini-Hochberg as in
  ``scipy.stats.false_discovery_control``.
* Dask and the grouping of rows give the same result; units, minimum number
  of values, datetime coordinates, Dataset input, wrong inputs.
"""
import numpy as np
import pandas as pd
import pytest

import xrMHW

xr = pytest.importorskip("xarray")
scipy_stats = pytest.importorskip("scipy.stats")

N_YEARS = 33
YEARS = np.arange(1993, 1993 + N_YEARS)


def _ar1(rng, m, n, phi):
    e = rng.normal(size=(m, n))
    x = np.empty((m, n))
    x[:, 0] = e[:, 0]
    for k in range(1, n):
        x[:, k] = phi * x[:, k - 1] + e[:, k]
    return x


@pytest.fixture(scope="module")
def field():
    """(lat=12, lon=25, year=33): trends of varied strength, AR(1) noise,
    integer-valued cells (ties, like event days) and missing years."""
    rng = np.random.default_rng(11)
    m = 12 * 25
    trend = rng.normal(0.0, 0.05, (m, 1)) * (YEARS - YEARS.mean())
    x = trend + _ar1(rng, m, N_YEARS, 0.6)
    x[:60] = np.round(3 * x[:60])                       # ties
    miss = rng.random((m, N_YEARS)) < 0.15
    miss[:100] = False                                  # complete rows
    x[miss] = np.nan
    x[200, :30] = np.nan                                # too few values
    x[201] = 5.0                                        # constant
    return xr.DataArray(x.reshape(12, 25, N_YEARS), dims=("lat", "lon", "year"),
                        coords={"year": YEARS, "lat": np.arange(12.0),
                                "lon": np.arange(25.0)},
                        name="mhw_days", attrs={"units": "days"})


def _rows(da):
    return da.values.reshape(-1, da.sizes["year"])


# ---------------------------------------------------------------- Sen + MK
def test_sen_and_mk_equal_pymannkendall(field):
    pmk = pytest.importorskip("pymannkendall")
    tr = xrMHW.xrMHW_trend(field, per="year")
    slope = tr["slope"].values.ravel()
    p = tr["p_value"].values.ravel()
    X = _rows(field)
    checked = 0
    for r in range(len(X)):
        if np.isfinite(X[r]).sum() < 3:
            assert np.isnan(slope[r]) and np.isnan(p[r])
            continue
        ref = pmk.original_test(X[r])
        assert np.isclose(slope[r], ref.slope, rtol=0, atol=1e-12), r
        assert np.isclose(p[r], ref.p, rtol=0, atol=1e-12), r
        checked += 1
    assert checked > 250


def test_hamed_rao_equals_pymannkendall_without_missing(field):
    pmk = pytest.importorskip("pymannkendall")
    tr = xrMHW.xrMHW_trend(field, test="hamed_rao", per="year")
    p = tr["p_value"].values.ravel()
    X = _rows(field)
    complete = np.flatnonzero(np.isfinite(X).all(axis=1))
    assert len(complete) >= 100
    differs = 0
    for r in complete:
        with np.errstate(invalid="ignore"):
            ref = pmk.hamed_rao_modification_test(X[r])
        assert np.isclose(p[r], ref.p, rtol=0, atol=1e-10, equal_nan=True), (
            r, p[r], ref.p)
        differs += not np.isclose(ref.p, pmk.original_test(X[r]).p)
    assert differs > 20, "the correction must change some p-values"


def _hamed_rao_reference(x, t, alpha=0.05):
    """pymannkendall's Hamed-Rao, written per cell, with the Sen trend
    removed at the true times of the valid values."""
    from scipy.stats import norm, rankdata
    ok = np.isfinite(x)
    xv, tv = x[ok], t[ok]
    n = len(xv)
    i, j = np.triu_indices(len(x), 1)
    pairs = ok[i] & ok[j]
    sen = np.median((x[j] - x[i])[pairs] / (t[j] - t[i])[pairs])
    s = sum(np.sign(xv[b] - xv[a]) for a in range(n) for b in range(a + 1, n))
    _, cnt = np.unique(xv, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(cnt * (cnt - 1) * (2 * cnt + 5))) / 18
    rk = rankdata(xv - sen * tv)
    y = rk - rk.mean()
    acf = np.correlate(y, y, "full")[n - 1:] / np.sum(y * y)
    bound = norm.ppf(1 - alpha / 2) / np.sqrt(n)
    sni = sum((n - k) * (n - k - 1) * (n - k - 2) * acf[k]
              for k in range(1, n) if abs(acf[k]) > bound)
    var *= 1 + 2 * sni / (n * (n - 1) * (n - 2))
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var)
    return 2 * (1 - norm.cdf(abs(z)))


def test_hamed_rao_with_missing_equals_direct_version(field):
    tr = xrMHW.xrMHW_trend(field, test="hamed_rao", per="year")
    p = tr["p_value"].values.ravel()
    X = _rows(field)
    t = (YEARS - YEARS[0] + 1).astype(float)          # positions 1..n
    gappy = [r for r in range(len(X))
             if 3 <= np.isfinite(X[r]).sum() < N_YEARS and np.nanstd(X[r]) > 0]
    assert len(gappy) > 100
    for r in gappy:
        with np.errstate(invalid="ignore"):
            ref = _hamed_rao_reference(X[r], t)
        assert np.isclose(p[r], ref, rtol=0, atol=1e-10, equal_nan=True), r


# ---------------------------------------------------------------- OLS
def test_ols_and_t_equal_linregress(field):
    tr = xrMHW.xrMHW_trend(field, method="ols", test="t", per="year")
    slope = tr["slope"].values.ravel()
    p = tr["p_value"].values.ravel()
    X = _rows(field)
    for r in range(len(X)):
        ok = np.isfinite(X[r])
        if ok.sum() < 3 or np.nanstd(X[r]) == 0:
            continue
        ref = scipy_stats.linregress(YEARS[ok], X[r, ok])
        assert np.isclose(slope[r], ref.slope, rtol=1e-10, atol=1e-12), r
        assert np.isclose(p[r], ref.pvalue, rtol=1e-8, atol=1e-12), r


def test_sen_slope_equals_theilslopes(field):
    tr = xrMHW.xrMHW_trend(field, test=None, per="year")
    assert "p_value" not in tr
    slope = tr["slope"].values.ravel()
    X = _rows(field)
    for r in range(0, len(X), 7):
        ok = np.isfinite(X[r])
        if ok.sum() < 3:
            continue
        ref = scipy_stats.theilslopes(X[r, ok], YEARS[ok]).slope
        assert np.isclose(slope[r], ref, rtol=0, atol=1e-12), r


# ---------------------------------------------------------------- FDR
def test_fdr_equals_benjamini_hochberg(field):
    if not hasattr(scipy_stats, "false_discovery_control"):
        pytest.skip("needs SciPy >= 1.11")
    tr = xrMHW.xrMHW_trend(field, fdr=0.10)
    p = tr["p_value"].values.ravel()
    ok = np.isfinite(p)
    adjusted = scipy_stats.false_discovery_control(p[ok], method="bh")
    expected = adjusted <= 0.10
    got = tr["significant_fdr"].values.ravel()
    assert np.array_equal(got[ok] == 1, expected)
    assert np.isnan(got[~ok]).all()
    assert 0 < expected.sum() <= (p[ok] < 0.10).sum()


# ---------------------------------------------------------------- engine
def test_dask_and_row_groups_equal_memory(field, monkeypatch):
    pytest.importorskip("dask")
    mem = xrMHW.xrMHW_trend(field, test="hamed_rao", fdr=0.1)
    chunked = field.chunk({"lat": 5, "lon": 10, "year": 11})
    assert xrMHW.xrMHW_trend(chunked, test="hamed_rao")["slope"].chunks is not None
    lazy = xrMHW.xrMHW_trend(chunked, test="hamed_rao", fdr=0.1)
    assert lazy["slope"].chunks is None        # fdr computes everything once
    monkeypatch.setattr(xrMHW, "_TREND_MAX_PAIRS", 2000)   # ~4 rows per group
    grouped = xrMHW.xrMHW_trend(field, test="hamed_rao", fdr=0.1)
    for v in mem.data_vars:
        assert np.array_equal(lazy[v].values, mem[v].values, equal_nan=True), v
        assert np.array_equal(grouped[v].values, mem[v].values, equal_nan=True), v


def test_units_min_valid_and_constant(field):
    dec = xrMHW.xrMHW_trend(field)
    yr = xrMHW.xrMHW_trend(field, per="year")
    assert np.allclose(dec["slope"], 10 * yr["slope"], equal_nan=True)
    assert dec["slope"].attrs["units"] == "days per decade"
    assert np.allclose(dec["p_value"], yr["p_value"], equal_nan=True)
    strict = xrMHW.xrMHW_trend(field, min_valid=30)
    n = field.notnull().sum("year")
    assert strict["slope"].where(n < 30).isnull().all()
    assert np.array_equal(strict["n_valid"].values, n.values)
    flat = dec.isel(lat=201 // 25, lon=201 % 25)              # constant cell
    assert float(flat["slope"]) == 0.0 and float(flat["p_value"]) == 1.0


def test_datetime_coordinate(field):
    dates = pd.to_datetime([f"{y}-07-02" for y in YEARS])
    by_date = xrMHW.xrMHW_trend(field.assign_coords(year=dates), per="year")
    by_year = xrMHW.xrMHW_trend(field, per="year")
    assert np.allclose(by_date["slope"], by_year["slope"], rtol=2e-3, equal_nan=True)
    assert np.allclose(by_date["p_value"], by_year["p_value"], equal_nan=True)


def test_dataset_of_annual_metrics():
    rng = np.random.default_rng(5)
    dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    doy = dates.dayofyear.values
    noise = _ar1(rng, 4, len(dates), 0.9)[..., None] * 0.35
    ramp = np.linspace(0, 1.5, len(dates))
    sst = (15 + 5 * np.sin(2 * np.pi * (doy - 80) / 365.25) + ramp)[None, :, None] + noise
    ds = xr.DataArray(sst.transpose(1, 0, 2), dims=("time", "lat", "lon"),
                      coords={"time": dates, "lat": np.arange(4.0), "lon": [0.0]},
                      name="sst").to_dataset()
    ann = xrMHW.xrMHW_annual(xrMHW.xrMHW_func(ds, "sst", (1991, 2020)))
    tr = xrMHW.xrMHW_trend(ann, fdr=0.1)
    assert {"mhw_days_slope", "mhw_count_p_value", "mhw_duration_n_valid",
            "mhw_days_significant_fdr"} <= set(tr.data_vars)
    assert (tr["mhw_days_slope"] > 0).all()          # a warming record
    assert tr["mhw_days_slope"].attrs["units"] == "days per decade"


# ---------------------------------------------------------------- confidence intervals
def test_sen_interval_equals_theilslopes(field):
    for alpha in (0.05, 0.10):
        tr = xrMHW.xrMHW_trend(field, per="year", alpha=alpha)
        lo, hi = tr["slope_low"].values.ravel(), tr["slope_high"].values.ravel()
        X = _rows(field)
        checked = 0
        for r in range(len(X)):
            ok = np.isfinite(X[r])
            if ok.sum() < 3:
                continue
            ref = scipy_stats.theilslopes(X[r, ok], YEARS[ok], alpha=1 - alpha)
            assert np.isclose(lo[r], ref.low_slope, rtol=0, atol=1e-12), (r, alpha)
            assert np.isclose(hi[r], ref.high_slope, rtol=0, atol=1e-12), (r, alpha)
            checked += 1
        assert checked > 250
    # the interval excludes zero about when the test rejects
    sig = tr["p_value"] < 0.10
    excl = (tr["slope_low"] > 0) | (tr["slope_high"] < 0)
    assert float((sig == excl).mean()) > 0.95


def test_ols_interval_equals_linregress(field):
    from scipy.stats import t as student_t
    tr = xrMHW.xrMHW_trend(field, method="ols", test="t", per="year")
    lo, hi = tr["slope_low"].values.ravel(), tr["slope_high"].values.ravel()
    X = _rows(field)
    for r in range(len(X)):
        ok = np.isfinite(X[r])
        if ok.sum() < 3 or np.nanstd(X[r]) == 0:
            continue
        ref = scipy_stats.linregress(YEARS[ok], X[r, ok])
        half = student_t.ppf(0.975, ok.sum() - 2) * ref.stderr
        assert np.isclose(lo[r], ref.slope - half, rtol=1e-9, atol=1e-12), r
        assert np.isclose(hi[r], ref.slope + half, rtol=1e-9, atol=1e-12), r
    # a significant t-test is exactly an interval that excludes zero
    excl = (tr["slope_low"] > 0) | (tr["slope_high"] < 0)
    assert bool(((tr["p_value"] < 0.05) == excl).where(tr["p_value"].notnull(), True).all())


def test_hamed_rao_interval_is_wider_with_positive_autocorrelation(field):
    mk = xrMHW.xrMHW_trend(field, per="year")
    hr = xrMHW.xrMHW_trend(field, test="hamed_rao", per="year")
    width_mk = mk["slope_high"] - mk["slope_low"]
    width_hr = hr["slope_high"] - hr["slope_low"]
    both = width_mk.notnull() & width_hr.notnull()
    assert float((width_hr >= width_mk - 1e-12).where(both).mean()) > 0.7
    assert hr["slope_low"].isnull().equals(hr["p_value"].isnull() | hr["slope"].isnull())


def test_lag_equals_pymannkendall(field):
    pmk = pytest.importorskip("pymannkendall")
    tr = xrMHW.xrMHW_trend(field, test="hamed_rao", lag=3, per="year")
    p = tr["p_value"].values.ravel()
    X = _rows(field)
    complete = np.flatnonzero(np.isfinite(X).all(axis=1))
    for r in complete:
        with np.errstate(invalid="ignore"):
            ref = pmk.hamed_rao_modification_test(X[r], lag=3).p
        assert np.isclose(p[r], ref, rtol=0, atol=1e-10, equal_nan=True), r
    # white noise: negative corrected variances (NaN p) are rarer with lag=3
    rng = np.random.default_rng(8)
    noise = xr.DataArray(rng.normal(size=(3000, N_YEARS)), dims=("cell", "year"),
                         coords={"year": YEARS})
    nan_all = int(xrMHW.xrMHW_trend(noise, test="hamed_rao")["p_value"].isnull().sum())
    nan_3 = int(xrMHW.xrMHW_trend(noise, test="hamed_rao", lag=3)["p_value"].isnull().sum())
    assert nan_3 < nan_all and nan_all > 10


# ---------------------------------------------------------------- FDR by level
def test_fdr_per_depth_level(field):
    if not hasattr(scipy_stats, "false_discovery_control"):
        pytest.skip("needs SciPy >= 1.11")
    rng = np.random.default_rng(4)
    deep = field.copy(data=rng.normal(size=field.shape))        # no trend at depth
    cube = xr.concat([field, deep], dim="depth")
    cube = cube.assign_coords(depth=[0.5, 100.0]).transpose("lat", "depth", "lon", "year")
    auto = xrMHW.xrMHW_trend(cube, fdr=0.10)
    assert auto.attrs["trend_fdr_by"] == "depth"
    whole = xrMHW.xrMHW_trend(cube, fdr=0.10, fdr_by=None)
    assert whole.attrs["trend_fdr_by"] == "whole field"
    assert auto["significant_fdr"].dims == auto["p_value"].dims
    for k in range(2):
        p = auto["p_value"].isel(depth=k).values.ravel()
        ok = np.isfinite(p)
        expected = scipy_stats.false_discovery_control(p[ok], method="bh") <= 0.10
        got = auto["significant_fdr"].isel(depth=k).values.ravel()
        assert np.array_equal(got[ok] == 1, expected), k
    p = whole["p_value"].values.ravel()
    ok = np.isfinite(p)
    expected = scipy_stats.false_discovery_control(p[ok], method="bh") <= 0.10
    assert np.array_equal(whole["significant_fdr"].values.ravel()[ok] == 1, expected)
    assert not auto["significant_fdr"].equals(whole["significant_fdr"])
    by_lat = xrMHW.xrMHW_trend(cube, fdr=0.10, fdr_by=["depth", "lat"])
    assert by_lat.attrs["trend_fdr_by"] == "depth, lat"


def test_long_series_warning_and_ols_without_pairs():
    rng = np.random.default_rng(2)
    da = xr.DataArray(rng.normal(size=(3, 2000)), dims=("cell", "time"),
                      coords={"time": pd.date_range("2000-01-01", periods=2000)})
    with pytest.warns(UserWarning, match="every pair"):
        xrMHW.xrMHW_trend(da.isel(cell=[0]), dim="time")
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        tr = xrMHW.xrMHW_trend(da, dim="time", method="ols", test="t", per="year")
    ref = scipy_stats.linregress(np.arange(2000) / 365.2425, da.values[0])
    assert np.isclose(float(tr["slope"][0]), ref.slope, rtol=1e-6)


def test_wrong_inputs(field):
    with pytest.raises(ValueError, match="lag"):
        xrMHW.xrMHW_trend(field, lag=3)
    with pytest.raises(ValueError, match="fdr_by"):
        xrMHW.xrMHW_trend(field, fdr=0.1, fdr_by="depth")
    with pytest.raises(ValueError, match="alpha"):
        xrMHW.xrMHW_trend(field, alpha=95)
    with pytest.raises(ValueError, match="method"):
        xrMHW.xrMHW_trend(field, method="median")
    with pytest.raises(ValueError, match="test"):
        xrMHW.xrMHW_trend(field, test="spearman")
    with pytest.raises(ValueError, match="ols"):
        xrMHW.xrMHW_trend(field, test="t")
    with pytest.raises(ValueError, match="fdr"):
        xrMHW.xrMHW_trend(field, test=None, fdr=0.1)
    with pytest.raises(ValueError, match="per"):
        xrMHW.xrMHW_trend(field, per="century")
    with pytest.raises(ValueError, match="time"):
        xrMHW.xrMHW_trend(field, dim="time")
