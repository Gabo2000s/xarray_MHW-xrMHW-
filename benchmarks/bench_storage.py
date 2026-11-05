"""
Storage cost of xrMHW output: format and compression measured separately
=========================================================================

The 22-Sep-2026 draft compared "Zarr" (1.1 GiB) with "uncompressed netCDF"
(15.6 GiB). Those two numbers came from different runs, and the comparison mixed
up the container format with compression. This script writes the SAME field,
with the SAME chunks, through several writers, so the two effects can be told
apart.

Design
------
1. Compute once. The input is chunked with chunk_spec(side=--chunk-side), the
   severity field (float32) is computed on a LocalCluster of worker processes
   (one thread each) and written to a source Zarr store. Its size, the codec
   Zarr actually used (read from the store metadata; zarr-python 2 defaults to
   Blosc/LZ4, zarr-python 3 to Zstd) and the elapsed time are recorded. This
   step includes the detection.
2. Write the copies. The cluster is closed and Dask's threaded scheduler is used,
   because writing netCDF from worker processes failed on large outputs. The
   source store is opened lazily with its own chunks and written, in turn, as

     a  netCDF-4, uncompressed       (zlib=False, chunksizes = Dask chunk shape)
     b  netCDF-4, zlib level 1       (shuffle=True, chunksizes = Dask chunk shape)
     c  Zarr, default compressor     (a second write, timed like a and b)
     d  Zarr, explicit Zstd          (only with --zstd-level)

   Each copy's size on disk (recursive for Zarr) and write time are recorded,
   and the copy is deleted right after it is measured unless --keep is given.
   The copies read the source store, which has just been written and is likely
   in the operating system's file cache.
3. Event fraction. The share of ocean cell-days that fall inside events (the
   non-NaN values of severity) explains the compression ratio.

Disk space: the script aborts if --outdir has less than 1.3 x (uncompressed size
+ source store) free; about 20 GiB for the full Grand Banks domain.

Usage
-----
    python bench_storage.py --input <GLORYS.nc> --var thetao --clim 1993 2022 \\
        --workers 14 --chunk-side 8 --outdir <folder with >= 20 GiB free>

    # quick test
    python bench_storage.py --input <GLORYS.nc> --subset-depth 4 --subset-side 12 \\
        --outdir <tmp> --results <tmp>

Results: bench_v3_storage_<host>.csv and .json in --results (default:
bench_v3/ next to this script).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xarray as xr

# Shares the provenance, subsetting and chunking code with the main benchmark, so
# both scripts chunk the field identically.
from benchmark_xrMHW_v3_3 import (PeakRSS, _remove, _size, _where, chunk_spec,
                                  provenance, subset)

import xrMHW  # noqa: E402

GiB = 1024 ** 3
VAR_OUT = "mhw_severity"


def zarr_codec(path, name):
    """Codec and chunking of one array, read from the store metadata."""
    import zarr
    a = zarr.open_group(str(path), mode="r")[name]
    meta = getattr(a, "metadata", None)
    comp = getattr(a, "compressors", None)
    if comp is None:
        comp = getattr(a, "compressor", None)
    return {"zarr_format": getattr(meta, "zarr_format", None) if meta is not None else 2,
            "compressors": repr(comp), "chunks": list(a.chunks)}


def zstd_encoding(level):
    import zarr
    if int(zarr.__version__.split(".")[0]) >= 3:
        from zarr.codecs import ZstdCodec
        return {"compressors": (ZstdCodec(level=level),)}
    from numcodecs import Zstd
    return {"compressor": Zstd(level=level)}


def check_free(folder, needed, what):
    free = shutil.disk_usage(folder).free
    print(f"  espacio: hace falta ~{needed / GiB:.1f} GiB ({what}), libre {free / GiB:.1f} GiB "
          f"en {folder}")
    if free < needed:
        sys.exit(f"Espacio insuficiente en {folder}: hacen falta ~{needed / GiB:.1f} GiB "
                 f"y hay {free / GiB:.1f} GiB. Usa --outdir en otro disco o un subconjunto.")


def clear_encoding(ds):
    ds = ds.copy()
    for v in ds.variables.values():
        v.encoding = {}
    return ds


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--var", default="thetao")
    ap.add_argument("--clim", nargs=2, type=int, default=[1993, 2022], metavar=("A", "B"))
    ap.add_argument("--workers", type=int, default=14,
                    help="worker processes (one thread each) for the compute step")
    ap.add_argument("--threads", type=int, default=None,
                    help="threads for the copies (threaded scheduler); default: physical cores")
    ap.add_argument("--chunk-side", type=int, default=8)
    ap.add_argument("--subset-depth", type=int, default=0, help="0 = all depth levels")
    ap.add_argument("--subset-side", type=int, default=0, help="0 = whole horizontal domain")
    ap.add_argument("--outdir", required=True,
                    help="where the stores and files are written (and deleted)")
    ap.add_argument("--results", default=str(here / "bench_v3"),
                    help="where the CSV and JSON go (default: bench_v3/ next to this script)")
    ap.add_argument("--zstd-level", type=int, default=None,
                    help="also write case d, Zarr with an explicit Zstd codec at this level")
    ap.add_argument("--keep", action="store_true", help="keep every store and file written")
    ap.add_argument("--allow-path-import", action="store_true")
    args = ap.parse_args()

    import dask
    from dask.distributed import Client, LocalCluster

    prov = provenance(args.allow_path_import)
    prov["script"] = Path(__file__).name
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    prov["input_path"], prov["input_drive"] = _where(args.input)
    prov["outdir"], prov["outdir_drive"] = _where(outdir)
    prov["results"], prov["results_drive"] = _where(results)
    threads = args.threads or psutil.cpu_count(logical=False) or 4
    prov["dask_threads"] = threads
    clim = tuple(args.clim)

    ds = xr.open_dataset(args.input)
    depth = args.subset_depth or ds.sizes.get("depth", 1)
    side = args.subset_side or max(ds.sizes.get("latitude", 1), ds.sizes.get("longitude", 1))
    sub = subset(ds, args.var, depth, side)
    spec = chunk_spec(sub, args.chunk_side)
    chunked = sub.chunk(spec)
    dims = [d for d in sub[args.var].dims]
    nt = sub.sizes["time"]
    cells = int(np.prod([sub.sizes[d] for d in dims if d != "time"]))
    label = "d" + "x".join(str(sub.sizes[d]) for d in ("depth", "latitude", "longitude")
                           if d in sub.sizes)
    uncompressed = cells * nt * 4                      # float32
    chunk_shape = tuple(c[0] for c in chunked[args.var].chunks)

    print("=" * 74)
    print(f" xrMHW {prov['xrMHW_version']} storage benchmark — {prov['host']}")
    print(f" domain {label}: {cells} cells x {nt} days; severity float32 = "
          f"{uncompressed / GiB:.2f} GiB uncompressed")
    print(f" chunks {dict(zip(dims, chunk_shape))}; zarr {prov['zarr']}, netCDF4 {prov['netCDF4']}")
    print(f" stores -> {prov['outdir']}   results -> {prov['results']}")
    print("=" * 74)
    check_free(outdir, 1.3 * (uncompressed + 0.15 * uncompressed),
               "copia sin comprimir + almacen fuente estimado")

    src = outdir / "_storage_source.zarr"
    _remove(src)
    rows, notes = [], []

    # 1. compute once, on worker processes, into the source store
    print(f"\n 1. deteccion + escritura del almacen fuente ({args.workers} procesos)")
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1, processes=True,
                           dashboard_address=None, silence_logs=50)
    client = Client(cluster)
    try:
        out = clear_encoding(xrMHW.xrMHW_func(chunked, args.var, clim, output_vars=["severity"]))
        with PeakRSS() as rss:
            t0 = time.perf_counter()
            out.to_zarr(src, mode="w")
            t_src = time.perf_counter() - t0
    finally:
        client.close()
        cluster.close()
    src_bytes = _size(src)
    codec = zarr_codec(src, VAR_OUT)
    rows.append(dict(case="source", writer="to_zarr default (computed on worker processes, "
                     "includes detection)", dask_scheduler="processes", workers=args.workers,
                     bytes=src_bytes, gib=src_bytes / GiB, seconds=t_src,
                     peak_rss_bytes=rss.delta, codec=codec["compressors"]))
    print(f"    {src_bytes / GiB:.3f} GiB en {t_src:.1f} s, codec {codec['compressors']}")

    check_free(outdir, 1.3 * uncompressed, "copia sin comprimir")

    # 2. the copies, threaded scheduler
    dask.config.set(scheduler="threads", num_workers=threads)
    source = xr.open_zarr(src, chunks={})
    # apply_ufunc moves the core dimension (time) last, so the output's dimension
    # order differs from the input's; take the netCDF chunk shape from the output.
    out_dims = source[VAR_OUT].dims
    out_chunks = dict(zip(out_dims, (c[0] for c in source[VAR_OUT].chunks)))
    if out_chunks != dict(zip(dims, chunk_shape)):
        notes.append(f"source store chunks {out_chunks} differ from the input chunks "
                     f"{dict(zip(dims, chunk_shape))}")
    nc_chunks = tuple(out_chunks[d] for d in out_dims)

    cases = [
        ("a", "to_netcdf, uncompressed", "netcdf",
         {"zlib": False, "chunksizes": nc_chunks}),
        ("b", "to_netcdf, zlib level 1 + shuffle", "netcdf",
         {"zlib": True, "complevel": 1, "shuffle": True, "chunksizes": nc_chunks}),
        ("c", "to_zarr, default compressor", "zarr", {}),
    ]
    if args.zstd_level is not None:
        cases.append(("d", f"to_zarr, Zstd level {args.zstd_level}", "zarr",
                      zstd_encoding(args.zstd_level)))

    for case, writer, fmt, enc in cases:
        target = outdir / (f"_storage_{case}.nc" if fmt == "netcdf" else f"_storage_{case}.zarr")
        _remove(target)
        dsw = clear_encoding(source)
        print(f"\n 2{case}. {writer}")
        try:
            t0 = time.perf_counter()
            if fmt == "netcdf":
                dsw.to_netcdf(target, engine="netcdf4", encoding={VAR_OUT: enc})
            else:
                dsw.to_zarr(target, mode="w", encoding={VAR_OUT: enc} if enc else None)
            t_w = time.perf_counter() - t0
            nbytes = _size(target)
            row = dict(case=case, writer=writer, dask_scheduler="threads", workers=threads,
                       bytes=nbytes, gib=nbytes / GiB, seconds=t_w, error=None)
            if fmt == "zarr":
                row["codec"] = zarr_codec(target, VAR_OUT)["compressors"]
            print(f"    {nbytes / GiB:.3f} GiB en {t_w:.1f} s")
        except Exception as exc:                               # noqa: BLE001
            row = dict(case=case, writer=writer, dask_scheduler="threads", workers=threads,
                       bytes=None, gib=None, seconds=None, error=f"{type(exc).__name__}: {exc}")
            print(f"    FALLO: {row['error'][:300]}")
        finally:
            if not args.keep:
                _remove(target)
        rows.append(row)

    # 3. event fraction
    print("\n 3. fraccion de dias-celda oceanicos dentro de eventos")
    n_event = int(source[VAR_OUT].notnull().sum().compute())
    first = sub[args.var].isel(time=0)
    n_ocean = int(first.notnull().sum().compute())
    event_fraction = n_event / (n_ocean * nt) if n_ocean else None
    print(f"    {n_event} valores no NaN de {n_ocean} celdas oceanicas x {nt} dias "
          f"= {100 * event_fraction:.2f}%")

    if not args.keep:
        _remove(src)

    a = next((r for r in rows if r["case"] == "a" and r["bytes"]), None)
    for r in rows:
        if a and r.get("bytes"):
            r["fraction_of_uncompressed"] = r["bytes"] / a["bytes"]
            r["uncompressed_over_this"] = a["bytes"] / r["bytes"]
        r.update(phase="storage", label=label, cells=cells, timesteps=nt,
                 chunks=str(out_chunks))

    by = {r["case"]: r for r in rows}
    summary = {
        "domain": label, "cells": cells, "timesteps": nt, "chunks": out_chunks,
        "uncompressed_float32_bytes_expected": uncompressed,
        "zarr_codec_default": codec,
        "event_cell_days": n_event, "ocean_cells": n_ocean,
        "event_fraction_of_ocean_cell_days": event_fraction,
        "cases": {k: {kk: v for kk, v in r.items()
                      if kk in ("writer", "bytes", "gib", "seconds", "codec", "error",
                                "fraction_of_uncompressed", "uncompressed_over_this")}
                  for k, r in by.items()},
    }
    if a and by.get("c", {}).get("bytes"):
        summary["ratio_nc_raw_over_zarr"] = a["bytes"] / by["c"]["bytes"]
    if a and by.get("b", {}).get("bytes"):
        summary["ratio_nc_raw_over_nc_zlib"] = a["bytes"] / by["b"]["bytes"]
    summary["provenance"] = prov
    summary["notes"] = notes + [
        "copies a-d read the source store, which had just been written and was probably "
        "in the OS file cache; their times are write times from a warm source"]

    host = prov["host"]
    pd.DataFrame(rows).to_csv(results / f"bench_v3_storage_{host}.csv", index=False)
    (results / f"bench_v3_storage_{host}.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print("\n" + "=" * 74)
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"},
                     indent=2, default=float))
    print(f"\n  -> {results / f'bench_v3_storage_{host}.csv'}")


if __name__ == "__main__":
    main()
