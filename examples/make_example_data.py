"""
Build the example datasets for the xrMHW notebooks
==================================================

Produces the two files that go into the separate Zenodo record the notebooks
download from, plus a registry of SHA-256 checksums for ``pooch``:

    xrmhw_glorys_4d_subset.nc   GLORYS12V1 potential temperature, 33 levels x
                                12 x 12 cells, daily, full record. Values are
                                written with the source packing (dtype,
                                scale_factor, add_offset), so they are identical
                                to what Copernicus serves.
    xrmhw_ostia_sst_3d.nc       OSTIA SST example already used by xrMHW v2,
                                losslessly recompressed (it is mostly land, so
                                compression shrinks it a lot). Units stay in
                                kelvin on purpose: converting them is one of the
                                steps the notebooks teach.
    registry.txt                "<file> sha256:<hash>" lines for pooch.

The window used for the notebooks is set with --lat0/--lon0 (its south-west
corner) on the south-eastern slope of the Grand Bank, so that it spans the shelf
edge: shallow columns on the shelf, full 33-level columns offshore. Without
them, the window with the most ocean cells at the deepest level is chosen.

Usage
-----
    python make_example_data.py \\
        --glorys E:\\A_Proyectos\\MHW_Grand_Banks\\Input\\GLORYS_GrandBanks_1993_2025.nc \\
        --ostia  D:\\Proyectos\\xarray_MHW-xrMHW-\\OSTIA_example_data.nc \\
        --out    E:\\A_Proyectos\\MHW_xrMHW_Paper\\example_data \\
        --lat0 43.5 --lon0 -49.5

The files are published at https://doi.org/10.5281/zenodo.22909109 under the
Copernicus Marine Service licence. The notebooks check the downloads against the
MD5 checksums of that record; rerunning this script produces files with the same
data but different bytes (the history attribute records the date), so a new upload
needs a new registry in the notebooks.
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

GLORYS_OUT = "xrmhw_glorys_4d_subset.nc"
OSTIA_OUT = "xrmhw_ostia_sst_3d.nc"
PACKING_KEYS = ("dtype", "scale_factor", "add_offset", "_FillValue", "missing_value")


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def spatial_dims(da):
    lat = next(d for d in da.dims if d.lower().startswith("lat"))
    lon = next(d for d in da.dims if d.lower().startswith("lon"))
    return lat, lon


def best_window(valid2d, side):
    """Index (i, j) of the side x side window with the most valid cells."""
    c = np.pad(valid2d.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    counts = c[side:, side:] - c[:-side, side:] - c[side:, :-side] + c[:-side, :-side]
    i, j = np.unravel_index(np.argmax(counts), counts.shape)
    return int(i), int(j), int(counts[i, j])


def packed_encoding(src_da, chunks, complevel=4):
    enc = {k: src_da.encoding[k] for k in PACKING_KEYS if k in src_da.encoding}
    enc.update(zlib=True, complevel=complevel, chunksizes=chunks)
    return enc


def build_glorys(path, var, out, side, lat0, lon0):
    ds = xr.open_dataset(path)
    da = ds[var]
    lat, lon = spatial_dims(da)
    depth = next((d for d in da.dims if d.lower().startswith("depth")), None)
    if depth is None:
        raise SystemExit(f"{var} has no depth dimension: {da.dims}")

    if lat0 is None or lon0 is None:
        bottom = da.isel(time=0, **{depth: -1}).load()
        i, j, n = best_window(np.isfinite(bottom.values), side)
        print(f"  window chosen automatically: {n}/{side * side} ocean cells at "
              f"{float(ds[depth][-1]):.1f} m")
    else:
        i = int(np.abs(ds[lat].values - lat0).argmin())
        j = int(np.abs(ds[lon].values - lon0).argmin())
    sub = ds[[var]].isel({lat: slice(i, i + side), lon: slice(j, j + side)})
    la, lo = sub[lat].values, sub[lon].values
    print(f"  latitude {la.min():.3f} to {la.max():.3f}, longitude {lo.min():.3f} to "
          f"{lo.max():.3f}, {sub.sizes[depth]} levels to {float(sub[depth][-1]):.1f} m, "
          f"{sub.sizes['time']} days ({str(sub.time.values[0])[:10]} to "
          f"{str(sub.time.values[-1])[:10]})")

    sub = sub.load()
    snap = sub[var].isel(time=0).notnull()
    valid = snap.mean((lat, lon)).values
    print(f"  ocean fraction per level: surface {valid[0]:.0%}, bottom {valid[-1]:.0%}")
    # deepest valid level of each column: tells whether the window spans the slope
    n_lev = snap.sum(depth).values.ravel()
    wet = n_lev[n_lev > 0]
    if wet.size:
        zb = sub[depth].values[wet - 1]
        print(f"  column bottoms: shallowest {zb.min():.0f} m, deepest {zb.max():.0f} m, "
              f"{(wet == sub.sizes[depth]).sum()}/{wet.size} columns reach the last level, "
              f"{side * side - wet.size} land cells")

    sub.attrs = {
        "title": "xrMHW example data: GLORYS12V1 4-D subset (Grand Banks slope)",
        "source": "E.U. Copernicus Marine Service Information (CMEMS), Global Ocean "
                  "Physics Reanalysis, GLOBAL_MULTIYEAR_PHY_001_030",
        "doi": "10.48670/moi-00021",
        "references": "Lellouche et al. (2021); https://github.com/Gabo2000s/xarray_MHW-xrMHW-",
        "subset": f"{side}x{side} cells, all levels, full record",
        "history": f"{datetime.now(timezone.utc):%Y-%m-%d} subset by make_example_data.py "
                   f"from {Path(path).name}; packing preserved",
    }
    # time contiguous in every chunk: the layout xrMHW reads fastest
    chunks = (sub.sizes["time"], 1, side, side)
    enc = {var: packed_encoding(ds[var], chunks)}
    sub.to_netcdf(out, encoding=enc)

    back = xr.open_dataset(out)[var].load()
    same = bool(((back.values == sub[var].values) | (back.isnull() & sub[var].isnull()).values).all())
    print(f"  written {out.stat().st_size / 1e6:.1f} MB; identical to the source subset: {same}")
    if not same:
        raise SystemExit("round-trip check failed")


def build_ostia(path, out):
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]
    ds = ds.load()
    lat, lon = spatial_dims(ds[var])
    chunks = (ds.sizes["time"], ds.sizes[lat], ds.sizes[lon])
    enc = {var: {"zlib": True, "complevel": 4, "chunksizes": chunks,
                 "dtype": ds[var].encoding.get("dtype", "float32"),
                 "_FillValue": ds[var].encoding.get("_FillValue", np.float32(np.nan))}}
    ds.attrs["history"] = (ds.attrs.get("history", "") + f"; {datetime.now(timezone.utc):%Y-%m-%d} "
                           "losslessly recompressed by make_example_data.py").lstrip("; ")
    ds.to_netcdf(out, encoding=enc)
    back = xr.open_dataset(out)[var].load()
    same = bool(((back.values == ds[var].values) | (back.isnull() & ds[var].isnull()).values).all())
    print(f"  {Path(path).stat().st_size / 1e6:.1f} MB -> {out.stat().st_size / 1e6:.1f} MB; "
          f"identical: {same}; units: {ds[var].attrs.get('units')}")
    if not same:
        raise SystemExit("round-trip check failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glorys", required=True)
    ap.add_argument("--ostia", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--var", default="thetao")
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--lat0", type=float, default=None)
    ap.add_argument("--lon0", type=float, default=None)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("GLORYS 4-D subset")
    build_glorys(a.glorys, a.var, out / GLORYS_OUT, a.side, a.lat0, a.lon0)
    print("OSTIA 3-D SST")
    build_ostia(a.ostia, out / OSTIA_OUT)

    lines = [f"{name} sha256:{sha256(out / name)}" for name in (GLORYS_OUT, OSTIA_OUT)]
    (out / "registry.txt").write_text("\n".join(lines) + "\n")
    print("\nregistry.txt\n  " + "\n  ".join(lines))
    print(f"\nUpload {GLORYS_OUT} and {OSTIA_OUT} to a new Zenodo record and send me its DOI.")


if __name__ == "__main__":
    main()
