"""
xrMHW benchmark, second design
==============================

The first design measured the out-of-core path with 11.5 MiB chunks inherited
from an older script, which is roughly forty times below the sensible size for
Dask. What it recorded was therefore not the package's out-of-core performance
but the cost of a bad chunking choice: hundreds of write tasks serialised
through the HDF5 lock. It also gave the in-memory path an unlimited memory
budget while constraining Dask to about 1 GB, so the two were never comparable.

This version fixes both and separates what the first conflated.

Phases (choose with --phase, default "all"; each writes its own CSV)
-------------------------------------------------------------------
scaling    In-memory ladder: throughput and the peak-RSS-to-raw amplification
           that sets the memory ceiling. This is the part of the first run
           that was sound; it is kept so the numbers are reproduced in one
           place.
sweep      Chunk-size sweep on one representative 4-D subset, with detection
           time and write time measured separately. Answers whether an optimum
           exists and where it is.
scheduler  Threads versus worker processes. The kernel is a per-cell Python
           loop with numpy sections that release the GIL, so threads
           parallelise it only partly (about 3x a single core on 14 threads in
           our runs); worker processes sidestep the GIL entirely. Give it at
           least as many chunks as workers, or idle workers distort the result.
ceiling    The decisive comparison: a domain too large for the in-memory path
           on this machine, processed out-of-core. Establishes that the
           limitation is real and that the out-of-core path removes it.
reference  marineHeatWaves (Oliver) and xmhw (coecms) on the SAME subset as
           xrMHW, at a size where fixed overheads do not dominate. The first
           run compared them on 16 cells and produced a 236x figure that is
           not defensible. xmhw's grid path breaks on recent xarray versions
           (its multi-index reassembly indexes a scalar coordinate); when that
           happens the script falls back to the same per-cell kernel, reusing
           the grid threshold, and records which path was timed.

Each phase writes its own CSV (bench_v2_<phase>_<host>.csv) plus a combined
file and one JSON summary, so phases can be run separately over several
sessions without overwriting each other.

Timing decomposition
--------------------
Detection is forced with a reduction over every output variable, which computes
the whole graph without writing. The write is then timed as a second pass that
recomputes detection and writes the result. Report the detection time and the
total of the second pass; treat their difference with caution. The first pass
reads the input from disk cold, and the operating system then caches it, so the
second pass reads it warm. The subtraction therefore underestimates the write,
sometimes grossly (we saw a 3.4 GiB NetCDF "written" in 0.7 s, and a full-domain
Zarr pass that finished faster than detection alone). Rows where the difference
is under 2% of the total are flagged write_resolved = False.

Writing (--write-format). With worker processes, NetCDF goes through a path in
which every chunk reopens the file in append mode under a distributed lock; on
the full domain one of those reopenings failed with "Unknown file format" after
5 GB had been written. Zarr writes one object per chunk, needs no cross-process
lock, and compresses by default, which matters because xrMHW's event fields are
NaN outside event days. A failed write no longer discards the detection time.

Peak memory includes child processes, so the worker-process scheduler is
measured fairly rather than only reporting the parent's resident set.

Usage
-----
    python benchmark_xrMHW_v3_2.py --input <GLORYS.nc> --var thetao \\
        --clim 1993 2022 --phase all

For marineHeatWaves, clone https://github.com/ecjoliver/marineHeatWaves and put
marineHeatWaves.py next to this script, or install it into the environment.
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import shutil
import socket
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xarray as xr

warnings.filterwarnings("ignore")

import xrMHW  # noqa: E402

BYTES_PER_MIB = 1024 ** 2


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def provenance(allow_path_import: bool) -> dict:
    import importlib.metadata as md

    origin = Path(xrMHW.__file__).resolve()
    mod_version = getattr(xrMHW, "__version__", None)
    try:
        dist_version = md.version("xrMHW")
    except md.PackageNotFoundError:
        dist_version = None

    editable = False
    if dist_version is not None:
        try:
            raw = md.distribution("xrMHW").read_text("direct_url.json")
            editable = bool(json.loads(raw).get("dir_info", {}).get("editable")) if raw else False
        except Exception:                                   # noqa: BLE001
            editable = False

    if dist_version is None:
        problem = f"xrMHW is not an installed distribution; imported from {origin}"
    elif mod_version != dist_version:
        problem = (f"module reports {mod_version} but pip recorded {dist_version}: "
                   f"a stray xrMHW.py at {origin} is shadowing the installed package")
    else:
        problem = None
    if problem and not allow_path_import:
        sys.exit(f"Refusing to benchmark: {problem}\n"
                 f"Install the package (pip install -e <repo>) and make sure no xrMHW.py "
                 f"sits in the working directory. --allow-path-import overrides this.")

    import dask
    import scipy
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gib": round(psutil.virtual_memory().total / 1024 ** 3, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "xrMHW_version": mod_version,
        "xrMHW_origin": str(origin),
        "xrMHW_install_mode": "editable" if editable else "regular",
        "numpy": np.__version__, "xarray": xr.__version__,
        "dask": dask.__version__, "scipy": scipy.__version__,
        "threads_pinned_to": os.environ.get("OMP_NUM_THREADS"),
    }


# ---------------------------------------------------------------------------
# Peak RSS, including child processes (needed for the process scheduler)
# ---------------------------------------------------------------------------

class PeakRSS:
    def __init__(self, interval=0.05, include_children=True):
        self.interval = interval
        self.include_children = include_children
        self._stop = threading.Event()

    def _sample(self):
        proc = psutil.Process()
        total = proc.memory_info().rss
        if self.include_children:
            for c in proc.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        return total

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._sample())
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.base = self.peak = self._sample()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def delta(self):
        return max(0, self.peak - self.base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subset(ds, var, n_depth, n_side):
    sel = {}
    for name, n in (("depth", n_depth), ("latitude", n_side), ("longitude", n_side)):
        if name in ds.dims:
            sel[name] = slice(0, n)
    return ds[[var]].isel(sel)


def shape_of(ds_sub, var):
    cells = int(np.prod([s for d, s in ds_sub.sizes.items() if d != "time"]))
    return cells, ds_sub.sizes["time"], ds_sub[var].nbytes


def chunk_spec(ds_sub, side, depth_whole=True):
    """Chunk over the horizontal only; depth is just another broadcast axis,
    and splitting it multiplies the task count for no benefit."""
    spec = {"time": -1}
    if "depth" in ds_sub.dims:
        spec["depth"] = -1 if depth_whole else min(ds_sub.sizes["depth"], side)
    for d in ("latitude", "longitude"):
        if d in ds_sub.dims:
            spec[d] = min(ds_sub.sizes[d], side)
    return spec


def check_space(outdir, needed_bytes, label):
    """Refuse to start a phase that would fill the disk halfway through.

    Every output variable is written at the same resolution as the input, so a
    run with all eight of them writes about eight times the raw array. The
    temporary file is deleted after each configuration, but it exists in full
    while the configuration runs.
    """
    free = shutil.disk_usage(outdir).free
    print(f"  escritura temporal maxima estimada: {needed_bytes / 1024**3:.1f} GiB, "
          f"libre en el destino: {free / 1024**3:.1f} GiB")
    if free < 1.2 * needed_bytes:
        sys.exit(f"Espacio insuficiente para la fase {label}: hace falta "
                 f"~{1.2 * needed_bytes / 1024**3:.1f} GiB libres en {outdir} y hay "
                 f"{free / 1024**3:.1f} GiB. Usa --outdir en otro disco, reduce el "
                 f"tamano del subconjunto, o limita las salidas con --output-vars.")


def n_chunks(ds_sub, spec):
    n = 1
    for d, s in ds_sub.sizes.items():
        c = spec.get(d, s)
        c = s if c in (-1, None) else c
        n *= int(np.ceil(s / c))
    return n


class WriteFailed(RuntimeError):
    """The detection pass finished but the write did not. Carries the detection
    time so a failed write does not throw away a valid measurement."""

    def __init__(self, t_detect, n_vars, error):
        super().__init__(error)
        self.t_detect, self.n_vars, self.error = t_detect, n_vars, error


def _remove(path):
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass                       # a worker may still hold a handle on Windows


def _size(path):
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


def time_detection_and_write(ds_chunked, var, clim, tmp, output_vars=None, fmt="netcdf"):
    """Detection time (reduction, no write), then total write; the write cost
    is the difference. Detection is recomputed in the second pass.

    Returns (t_detect, t_total, t_write, written_bytes, n_vars, write_reliable).
    When the second pass happens to be faster than the first, the subtraction
    yields a negative write time; it is clamped to zero and flagged, because in
    that case the run-to-run scatter exceeds the quantity being measured.

    fmt: "netcdf", "zarr", or "none" (detection only; total = detection).
    Raises WriteFailed, carrying the detection time, if the write fails.
    """
    import dask

    out = xrMHW.xrMHW_func(ds_chunked, var, clim, output_vars=output_vars)
    n_vars = len(out.data_vars)
    t0 = time.perf_counter()
    dask.compute(*[v.mean() for v in out.data_vars.values()])
    t_detect = time.perf_counter() - t0

    if fmt == "none":
        return t_detect, t_detect, 0.0, 0, n_vars, False

    out = xrMHW.xrMHW_func(ds_chunked, var, clim, output_vars=output_vars)
    target = tmp.with_suffix(".zarr") if fmt == "zarr" else tmp
    _remove(target)
    try:
        t0 = time.perf_counter()
        if fmt == "zarr":
            # Encodings inherited from the NetCDF source (chunksizes, zlib, ...)
            # are not zarr encodings; let xarray choose fresh ones.
            for v in out.variables.values():
                v.encoding = {}
            out.to_zarr(target, mode="w")
        else:
            out.to_netcdf(target)
        t_total = time.perf_counter() - t0
        written = _size(target)
    except Exception as exc:                                   # noqa: BLE001
        raise WriteFailed(t_detect, n_vars, f"{type(exc).__name__}: {exc}") from exc
    finally:
        _remove(target)
    t_write = t_total - t_detect
    return (t_detect, t_total, max(0.0, t_write), written,
            n_vars, bool(t_write > 0.02 * t_total))


# ---------------------------------------------------------------------------
# Phase: in-memory scaling
# ---------------------------------------------------------------------------

def phase_scaling(ds, var, clim, outdir, prov, budget_frac):
    cases = [(1, 12), (1, 24), (2, 24), (4, 24), (8, 24), (16, 24), (33, 16), (33, 24), (33, 36)]
    budget = budget_frac * psutil.virtual_memory().total
    rows, seen = [], set()
    print("\n" + "=" * 74)
    print(" PHASE scaling — in-memory ladder (throughput and memory amplification)")
    print("=" * 74)
    for nd, ns in cases:
        sub = subset(ds, var, nd, ns)
        cells, nt, raw = shape_of(sub, var)
        label = f"d{nd}x{ns}x{ns}"
        # A requested subset larger than the file is clamped by isel, so two
        # cases can collapse onto the same array. Measuring it twice would put
        # duplicate points into the memory fit and bias its slope.
        shape = tuple(sub.sizes[d] for d in sorted(sub.sizes))
        if shape in seen:
            print(f"  {label:12s} skipped, identical to an earlier case after clamping")
            continue
        seen.add(shape)
        if 7.5 * raw > budget:
            print(f"  {label:12s} skipped, projected peak {7.5 * raw / 1024**3:.1f} GiB > budget")
            continue
        loaded = sub.load()
        with PeakRSS() as rss:
            t0 = time.perf_counter()
            out = xrMHW.xrMHW_func(loaded, var, clim)
            _ = out["mhw_intensity"].values
            elapsed = time.perf_counter() - t0
        del out, loaded
        rows.append(dict(phase="scaling", label=label, cells=cells, timesteps=nt,
                         raw_bytes=raw, seconds=elapsed,
                         cell_steps_per_s=cells * nt / elapsed,
                         peak_rss_bytes=rss.delta, peak_over_raw=rss.delta / raw))
        print(f"  {label:12s} {raw / BYTES_PER_MIB:8.1f} MiB  {elapsed:7.1f} s  "
              f"{cells * nt / elapsed / 1e3:7.0f} k cell-steps/s  peak {rss.delta / raw:5.2f}x")
    return rows


# ---------------------------------------------------------------------------
# Phase: chunk sweep
# ---------------------------------------------------------------------------

def phase_sweep(ds, var, clim, outdir, prov, depth, side, sides, output_vars=None,
                threads=1, amplification=7.26, budget_frac=0.75, fmt="netcdf"):
    sub = subset(ds, var, depth, side)
    cells, nt, raw = shape_of(sub, var)
    tmp = outdir / "_sweep.nc"
    budget = budget_frac * psutil.virtual_memory().total
    rows = []
    print("\n" + "=" * 74)
    print(f" PHASE sweep — chunk size on d{depth}x{side}x{side} "
          f"({raw / 1024**3:.2f} GiB raw, {cells} cells)")
    print("=" * 74)
    check_space(outdir, raw * (len(output_vars) if output_vars else 8), "sweep")
    print(f"  presupuesto de memoria: {budget / 1024**3:.0f} GiB "
          f"({budget_frac:.0%} de la RAM), {threads} hilos concurrentes")
    print(f"  {'chunk side':>10} {'chunks':>7} {'MiB/chunk':>10} {'proy GiB':>9} {'detect':>8} "
          f"{'write':>8} {'total':>8} {'peak GiB':>9}")
    seen_counts = set()
    for cs in sides:
        if cs > side:
            continue
        spec = chunk_spec(sub, cs)
        nch = n_chunks(sub, spec)
        # Two chunk sides that split the domain identically measure the same thing.
        if nch in seen_counts:
            print(f"  {cs:>10} {nch:>7}   identico al reparto anterior, omitido")
            continue
        seen_counts.add(nch)
        per_chunk = raw / nch
        # Peak scales with the number of chunks resident at once, which for the
        # threaded scheduler is the thread count. This is the relation that makes
        # an optimum exist: bigger chunks cut task overhead but raise the peak.
        projected = min(threads, nch) * per_chunk * amplification
        if projected > budget:
            print(f"  {cs:>10} {nch:>7} {per_chunk / BYTES_PER_MIB:>10.1f} "
                  f"{projected / 1024**3:>9.1f}   omitido: proyeccion sobre el presupuesto")
            rows.append(dict(phase="sweep", label=f"d{depth}x{side}x{side}", chunk_side=cs,
                             n_chunks=nch, mib_per_chunk=per_chunk / BYTES_PER_MIB,
                             projected_peak_bytes=projected, skipped_over_budget=True))
            continue
        chunked = sub.chunk(spec)
        with PeakRSS() as rss:
            t_d, t_tot, t_w, written, nvar, ok = time_detection_and_write(
                chunked, var, clim, tmp, output_vars, fmt)
        rows.append(dict(phase="sweep", label=f"d{depth}x{side}x{side}", chunk_side=cs,
                         n_chunks=nch, mib_per_chunk=per_chunk / BYTES_PER_MIB,
                         projected_peak_bytes=projected, skipped_over_budget=False,
                         cells=cells, timesteps=nt, raw_bytes=raw,
                         detect_s=t_d, write_s=t_w, write_resolved=ok, total_s=t_tot,
                         peak_rss_bytes=rss.delta, written_bytes=written,
                         cell_steps_per_s=cells * nt / t_tot))
        print(f"  {cs:>10} {nch:>7} {per_chunk / BYTES_PER_MIB:>10.1f} "
              f"{projected / 1024**3:>9.1f} {t_d:>8.1f} {t_w:>8.1f} "
              f"{t_tot:>8.1f} {rss.delta / 1024**3:>9.2f}")
    measured = [r for r in rows if not r["skipped_over_budget"]]
    if measured:
        best = min(measured, key=lambda r: r["total_s"])
        print(f"\n  optimum at chunk side {best['chunk_side']} "
              f"({best['mib_per_chunk']:.0f} MiB/chunk, {best['total_s']:.1f} s); "
              f"worst/best = {max(r['total_s'] for r in measured) / best['total_s']:.1f}x")
        err = [abs(r["peak_rss_bytes"] - r["projected_peak_bytes"]) / r["peak_rss_bytes"]
               for r in measured if r["peak_rss_bytes"] > 0]
        if err:
            print(f"  error medio del modelo de memoria "
                  f"(hilos x chunk x {amplification:.2f}): {100 * sum(err) / len(err):.0f}%")
    return rows


# ---------------------------------------------------------------------------
# Phase: scheduler
# ---------------------------------------------------------------------------

def phase_scheduler(ds, var, clim, outdir, prov, depth, side, chunk_side, worker_counts,
                    output_vars=None, fmt="netcdf"):
    sub = subset(ds, var, depth, side)
    cells, nt, raw = shape_of(sub, var)
    spec = chunk_spec(sub, chunk_side)
    chunked = sub.chunk(spec)
    tmp = outdir / "_sched.nc"
    rows = []
    print("\n" + "=" * 74)
    print(f" PHASE scheduler — threads vs processes, d{depth}x{side}x{side}, "
          f"chunk side {chunk_side} ({n_chunks(sub, spec)} chunks), escritura: {fmt}")
    print("=" * 74)
    if fmt != "none":
        check_space(outdir, raw * (len(output_vars) if output_vars else 8), "scheduler")

    def run(name, workers):
        error = None
        with PeakRSS() as rss:
            try:
                t_d, t_tot, t_w, written, _, ok = time_detection_and_write(
                    chunked, var, clim, tmp, output_vars, fmt)
            except WriteFailed as wf:
                t_d, t_tot, t_w, written, ok, error = wf.t_detect, None, None, 0, False, wf.error
        row = dict(phase="scheduler", scheduler=name, workers=workers, write_format=fmt,
                   write_error=error, detect_s=t_d, write_s=t_w, write_resolved=ok,
                   total_s=t_tot, cells=cells, timesteps=nt, written_bytes=written,
                   raw_bytes=raw, peak_rss_bytes=rss.delta,
                   detect_cell_steps_per_s=cells * nt / t_d,
                   cell_steps_per_s=(cells * nt / t_tot) if t_tot else None)
        tail = (f"write {t_w:7.1f}  total {t_tot:7.1f}" if t_tot is not None
                else "write FALLIDA (deteccion conservada)")
        print(f"  {name + (f' x{workers}' if workers else ''):<26} detect {t_d:7.1f}  {tail}  "
              f"peak {rss.delta / 1024**3:5.2f} GiB  "
              f"({cells * nt / t_d / 1e6:5.2f} M cell-steps/s)")
        if error:
            print(f"      error de escritura: {error[:200]}")
        return row

    rows.append(run("threads (default)", None))

    try:
        from dask.distributed import Client, LocalCluster
    except ImportError:
        print("  dask.distributed not installed; process scheduler skipped "
              "(conda install -c conda-forge distributed)")
        return rows

    for nw in worker_counts:
        cluster = client = None
        try:
            cluster = LocalCluster(n_workers=nw, threads_per_worker=1, processes=True,
                                   dashboard_address=None, silence_logs=50)
            client = Client(cluster)
            rows.append(run("processes", nw))
        except Exception as exc:                              # noqa: BLE001
            print(f"  processes x{nw}: failed ({type(exc).__name__}: {exc})")
        finally:
            if client is not None:
                client.close()
            if cluster is not None:
                cluster.close()
    return rows


# ---------------------------------------------------------------------------
# Phase: ceiling
# ---------------------------------------------------------------------------

def phase_ceiling(ds, var, clim, outdir, prov, amplification, chunk_side, side,
                  output_vars=("severity",), workers=0, fmt="netcdf"):
    total_ram = psutil.virtual_memory().total
    depth = ds.sizes.get("depth", 1)
    side = side or max(ds.sizes.get("latitude", 1), ds.sizes.get("longitude", 1))
    sub = subset(ds, var, depth, side)
    cells, nt, raw = shape_of(sub, var)
    projected = amplification * raw
    spec = chunk_spec(sub, chunk_side)
    tmp = outdir / "_ceiling.nc"
    output_vars = list(output_vars) if output_vars else None

    print("\n" + "=" * 74)
    print(" PHASE ceiling — a domain the in-memory path cannot hold")
    print("=" * 74)
    dims = "x".join(str(sub.sizes[d]) for d in ("depth", "latitude", "longitude") if d in sub.sizes)
    print(f"  domain d{dims}: {cells} cells, {raw / 1024**3:.2f} GiB raw")
    print(f"  salidas escritas: {', '.join(output_vars) if output_vars else 'todas (8)'}")
    print(f"  in-memory would need ~{projected / 1024**3:.0f} GiB "
          f"({amplification:.2f}x raw) against {total_ram / 1024**3:.0f} GiB installed"
          f"  -> {'DOES NOT FIT' if projected > total_ram else 'fits'}")
    print(f"  in-memory ceiling on this machine: "
          f"{total_ram / amplification / 1024**3:.1f} GiB of input")

    if fmt != "none":
        check_space(outdir, raw * (len(output_vars) if output_vars else 8), "ceiling")
    chunked = sub.chunk(spec)
    label = "d" + "x".join(str(sub.sizes[d]) for d in ("depth", "latitude", "longitude")
                           if d in sub.sizes)

    cluster = client = None
    if workers:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=workers, threads_per_worker=1, processes=True,
                               dashboard_address=None, silence_logs=50)
        client = Client(cluster)
        print(f"  planificador: {workers} procesos trabajadores, "
              f"{n_chunks(sub, spec)} chunks, escritura: {fmt}")
    else:
        print(f"  planificador: hilos (por defecto), {n_chunks(sub, spec)} chunks, "
              f"escritura: {fmt}")
    error = None
    try:
        with PeakRSS() as rss:
            try:
                t_d, t_tot, t_w, written, nvar, ok = time_detection_and_write(
                    chunked, var, clim, tmp, output_vars, fmt)
            except WriteFailed as wf:
                t_d, t_tot, t_w, written, nvar, ok = wf.t_detect, None, None, 0, wf.n_vars, False
                error = wf.error
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

    print(f"  deteccion: {t_d:.1f} s = {t_d / 60:.1f} min, "
          f"{cells * nt / t_d / 1e6:.2f} M cell-steps/s, pico {rss.delta / 1024**3:.2f} GiB")
    if error:
        print(f"  ESCRITURA FALLIDA ({fmt}); la deteccion se conserva. Error: {error[:300]}")
    elif fmt != "none":
        print(f"  escritura: {t_w:.1f} s, total {t_tot / 60:.1f} min, "
              f"escribio {written / 1024**3:.2f} GiB")
    return [dict(phase="ceiling", label=label, workers=workers or None,
                 write_format=fmt, write_error=error,
                 cells=cells, timesteps=nt,
                 raw_bytes=raw, projected_in_memory_bytes=projected,
                 fits_in_memory=bool(projected <= total_ram), chunk_side=chunk_side,
                 n_chunks=n_chunks(sub, spec), detect_s=t_d, write_s=t_w,
                 write_resolved=ok, total_s=t_tot, n_output_vars=nvar,
                 peak_rss_bytes=rss.delta, written_bytes=written,
                 peak_over_raw=rss.delta / raw,
                 detect_cell_steps_per_s=cells * nt / t_d,
                 cell_steps_per_s=(cells * nt / t_tot) if t_tot else None)]


# ---------------------------------------------------------------------------
# Phase: reference implementations, same subset for all three
# ---------------------------------------------------------------------------

def phase_reference(ds, var, clim, outdir, prov, side):
    sub = subset(ds, var, 1, side)
    cells, nt, raw = shape_of(sub, var)
    loaded = sub.load()
    rows, notes = [], []
    print("\n" + "=" * 74)
    print(f" PHASE reference — same subset for all three: d1x{side}x{side} "
          f"({cells} cells, {raw / BYTES_PER_MIB:.1f} MiB)")
    print("=" * 74)

    t0 = time.perf_counter()
    out = xrMHW.xrMHW_func(loaded, var, clim)
    _ = out["mhw_intensity"].values
    t_xr = time.perf_counter() - t0
    del out
    rows.append(dict(phase="reference", tool="xrMHW (in-memory)", cells=cells, timesteps=nt,
                     seconds=t_xr, cell_steps_per_s=cells * nt / t_xr))
    print(f"  {'xrMHW':<28} {t_xr:8.1f} s   {cells * nt / t_xr / 1e3:8.0f} k cell-steps/s")

    try:
        import marineHeatWaves as ref
        if not hasattr(np, "NaN"):
            np.NaN = np.nan
        flat = loaded[var].stack(cell=[d for d in loaded[var].dims if d != "time"])
        t_ord = np.array([d.toordinal() for d in loaded.indexes["time"].to_pydatetime()])
        t0, n_ok = time.perf_counter(), 0
        for k in range(flat.sizes["cell"]):
            ts = np.asarray(flat.isel(cell=k).values, dtype=float)
            if np.isnan(ts).all():
                continue
            ref.detect(t_ord, ts.copy(), climatologyPeriod=list(clim))
            n_ok += 1
        t_ref = time.perf_counter() - t0
        rows.append(dict(phase="reference", tool="marineHeatWaves (cell by cell)", cells=n_ok,
                         timesteps=nt, seconds=t_ref, cell_steps_per_s=n_ok * nt / t_ref))
        print(f"  {'marineHeatWaves':<28} {t_ref:8.1f} s   {n_ok * nt / t_ref / 1e3:8.0f} "
              f"k cell-steps/s   ({t_ref / t_xr:.2f}x xrMHW, {n_ok} ocean cells)")
    except ImportError:
        notes.append("marineHeatWaves not importable: clone "
                     "https://github.com/ecjoliver/marineHeatWaves and put marineHeatWaves.py "
                     "next to this script")
        print("  marineHeatWaves            not importable — see note in the JSON")

    try:
        from xmhw.xmhw import detect as xmhw_detect, threshold as xmhw_threshold

        # xmhw expects the horizontal grid only; a length-one depth dimension
        # survives `threshold` but breaks the reassembly inside `detect`.
        da = loaded[var]
        da = da.squeeze(drop=True) if "depth" in da.dims else da

        t0 = time.perf_counter()
        clim_ds = xmhw_threshold(da, climatologyPeriod=list(clim)).compute()
        t_thresh = time.perf_counter() - t0

        # Preferred: the grid path, which is how xmhw is meant to be used.
        # Recent xarray versions break its multi-index reassembly (it indexes a
        # scalar cell coordinate), so fall back to calling the same per-cell
        # kernel one cell at a time, reusing the grid threshold. The work done
        # is the same; only the assembly of the result differs.
        path, t0 = "grid", time.perf_counter()
        try:
            res = xmhw_detect(da, clim_ds.thresh, clim_ds.seas)
            _ = res.compute() if hasattr(res, "compute") else res
        except Exception as exc:                               # noqa: BLE001
            notes.append(f"xmhw grid path failed ({type(exc).__name__}: {exc}); "
                         f"timed the per-cell path with the grid threshold instead")
            path, t0 = "per-cell (grid path unavailable)", time.perf_counter()
            hdims = [d for d in da.dims if d != "time"]
            idx = np.ndindex(*[da.sizes[d] for d in hdims])
            for pos in idx:
                sel = dict(zip(hdims, pos))
                if bool(np.isnan(da.isel(sel).values).all()):
                    continue
                xmhw_detect(da.isel(sel), clim_ds.thresh.isel(sel), clim_ds.seas.isel(sel))
        t_det = time.perf_counter() - t0
        t_xm = t_thresh + t_det

        rows.append(dict(phase="reference", tool=f"xmhw [{path}]", cells=cells, timesteps=nt,
                         seconds=t_xm, threshold_s=t_thresh, detect_s=t_det,
                         cell_steps_per_s=cells * nt / t_xm))
        print(f"  {'xmhw':<28} {t_xm:8.1f} s   {cells * nt / t_xm / 1e3:8.0f} k cell-steps/s   "
              f"({t_xm / t_xr:.2f}x xrMHW; threshold {t_thresh:.1f} s + detect {t_det:.1f} s, "
              f"{path})")
    except Exception as exc:                                   # noqa: BLE001
        notes.append(f"xmhw not usable: {type(exc).__name__}: {exc}")
        print(f"  xmhw                       not usable ({type(exc).__name__}: {exc})")

    return rows, notes


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--var", default="thetao")
    ap.add_argument("--clim", nargs=2, type=int, default=[1993, 2022], metavar=("A", "B"))
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--phase", default="all",
                    choices=["all", "scaling", "sweep", "scheduler", "ceiling", "reference"])
    ap.add_argument("--sweep-depth", type=int, default=33)
    ap.add_argument("--sweep-side", type=int, default=24)
    ap.add_argument("--sweep-chunks", type=int, nargs="+", default=[4, 6, 8, 12, 16, 24])
    ap.add_argument("--chunk-side", type=int, default=None,
                    help="chunk side for the scheduler and ceiling phases; "
                         "defaults to the sweep optimum, or 12 if the sweep did not run")
    ap.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--output-vars", nargs="+", default=None,
                    help="restrict the written outputs in the sweep and scheduler phases; "
                         "the default writes all eight, which is the realistic case")
    ap.add_argument("--ceiling-output-vars", nargs="+", default=["severity"],
                    help="outputs written in the ceiling phase (default: severity only). "
                         "All eight would write about eight times the raw domain, which for "
                         "a full-domain run is tens of gigabytes; pass 'all' to force it.")
    ap.add_argument("--write-format", choices=["netcdf", "zarr", "none"], default="netcdf",
                    help="how the sweep, scheduler and ceiling phases write their output. "
                         "'none' times detection only. With worker processes, NetCDF goes "
                         "through a reopen-per-chunk path that failed on the full domain; "
                         "zarr writes one file per chunk and needs no cross-process lock "
                         "(conda install -c conda-forge zarr).")
    ap.add_argument("--ceiling-workers", type=int, default=0,
                    help="run the ceiling phase on this many worker processes; 0 (default) "
                         "uses the threaded scheduler. The kernel is GIL-bound, so the "
                         "process scheduler is the configuration worth reporting.")
    ap.add_argument("--ceiling-side", type=int, default=0,
                    help="horizontal side for the ceiling demonstration; 0 uses the whole "
                         "horizontal domain, which is the largest case the file allows")
    ap.add_argument("--ref-side", type=int, default=16,
                    help="horizontal side of the subset shared by all three implementations")
    ap.add_argument("--threads", type=int, default=None,
                    help="threads for Dask's default scheduler; defaults to the physical "
                         "core count. Peak memory scales with this times the chunk size, "
                         "so leaving it at the logical core count can exhaust RAM.")
    ap.add_argument("--amplification", type=float, default=None,
                    help="peak-RSS-to-raw ratio used by the sweep's memory model and by the "
                         "ceiling projection when the scaling phase is not run in the same "
                         "invocation (default: the measured value, else 7.26)")
    ap.add_argument("--budget-frac", type=float, default=0.6)
    ap.add_argument("--sweep-budget-frac", type=float, default=0.75,
                    help="fraction of RAM a sweep configuration may be projected to need "
                         "before it is skipped instead of measured")
    ap.add_argument("--min-fit-mib", type=float, default=64.0)
    ap.add_argument("--allow-path-import", action="store_true")
    args = ap.parse_args()

    prov = provenance(args.allow_path_import)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    clim = tuple(args.clim)
    ds = xr.open_dataset(args.input)

    import dask
    threads = args.threads or psutil.cpu_count(logical=False) or 4
    dask.config.set(num_workers=threads)
    prov["dask_threads"] = threads

    print("=" * 74)
    print(f" xrMHW {prov['xrMHW_version']} benchmark v2 — {prov['host']}, {prov['cpu'][:34]}")
    print(f" {prov['physical_cores']} physical cores, {prov['total_ram_gib']} GiB RAM, "
          f"{prov['threads_pinned_to']} BLAS thread pinned, {threads} Dask threads")
    print(f" imported from {prov['xrMHW_origin']} ({prov['xrMHW_install_mode']})")
    print("=" * 74)

    want = args.phase
    rows, notes, summary, by_phase = [], [], {}, {}

    def keep(name, r):
        by_phase[name] = r
        rows.extend(r)

    amplification = args.amplification or 7.26   # fallback if scaling is not run here
    if want in ("all", "scaling"):
        r = phase_scaling(ds, args.var, clim, outdir, prov, args.budget_frac)
        keep("scaling", r)
        fit = [x for x in r if x["raw_bytes"] >= args.min_fit_mib * BYTES_PER_MIB]
        if len(fit) >= 3:
            xs = np.array([x["raw_bytes"] for x in fit])
            ys = np.array([x["peak_rss_bytes"] for x in fit])
            slope, intercept = np.polyfit(xs, ys, 1)
            amplification = float(slope)
            tp = np.array([x["cell_steps_per_s"] for x in fit])
            summary["in_memory"] = {
                "amplification_peak_over_raw": amplification,
                "intercept_bytes": float(intercept),
                "r_squared": float(np.corrcoef(xs, ys)[0, 1] ** 2),
                "fit_n_points": len(fit),
                "throughput_mean_cell_steps_per_s": float(tp.mean()),
                "throughput_cv_percent_fitted_subsets": float(100 * tp.std(ddof=1) / tp.mean()),
                "ceiling_input_gib": float(psutil.virtual_memory().total / amplification / 1024 ** 3),
            }
            print(f"\n  amplification {amplification:.2f}x raw (R2 {summary['in_memory']['r_squared']:.5f}, "
                  f"n={len(fit)}); ceiling {summary['in_memory']['ceiling_input_gib']:.1f} GiB of input")

    best_chunk = args.chunk_side or 12
    if want in ("all", "sweep"):
        r = phase_sweep(ds, args.var, clim, outdir, prov,
                        args.sweep_depth, args.sweep_side, args.sweep_chunks,
                        args.output_vars, threads, amplification, args.sweep_budget_frac,
                        args.write_format)
        keep("sweep", r)
        r = [x for x in r if not x["skipped_over_budget"]]
        if r:
            b = min(r, key=lambda x: x["total_s"])
            best_chunk = args.chunk_side or b["chunk_side"]
            summary["sweep"] = {"best_chunk_side": b["chunk_side"],
                                "best_mib_per_chunk": b["mib_per_chunk"],
                                "best_total_s": b["total_s"],
                                "worst_over_best": max(x["total_s"] for x in r) / b["total_s"],
                                "write_fraction_at_best": b["write_s"] / b["total_s"]}

    if want in ("all", "scheduler"):
        r = phase_scheduler(ds, args.var, clim, outdir, prov,
                            args.sweep_depth, args.sweep_side, best_chunk, args.workers,
                            args.output_vars, args.write_format)
        keep("scheduler", r)
        # compare on detection, which is always measured; totals may be missing
        # when a write failed
        thr = next((x for x in r if x["workers"] is None), None)
        proc = [x for x in r if x["workers"] is not None]
        if thr and proc:
            fd = min(proc, key=lambda x: x["detect_s"])
            summary["scheduler_detection"] = {
                "threads_detect_s": thr["detect_s"],
                "best_processes_workers": fd["workers"],
                "best_processes_detect_s": fd["detect_s"],
                "process_detect_speedup_over_threads": thr["detect_s"] / fd["detect_s"],
            }
        thr = thr if (thr and thr["total_s"]) else None
        proc = [x for x in proc if x["total_s"]]
        if thr and proc:
            fastest = min(proc, key=lambda x: x["total_s"])
            summary["scheduler"] = {
                "threads_total_s": thr["total_s"],
                "best_processes_workers": fastest["workers"],
                "best_processes_total_s": fastest["total_s"],
                "process_speedup_over_threads": thr["total_s"] / fastest["total_s"],
                "parallel_claim_supported": bool(thr["total_s"] / fastest["total_s"] > 1.3),
            }

    if want in ("all", "ceiling"):
        cov = args.ceiling_output_vars
        cov = None if (cov and cov[0].lower() == "all") else cov
        keep("ceiling", phase_ceiling(ds, args.var, clim, outdir, prov,
                                      amplification, best_chunk, args.ceiling_side, cov,
                                      args.ceiling_workers, args.write_format))

    if want in ("all", "reference"):
        r, n = phase_reference(ds, args.var, clim, outdir, prov, args.ref_side)
        keep("reference", r)
        notes += n
        base = next((x for x in r if x["tool"].startswith("xrMHW")), None)
        if base:
            summary["reference"] = {
                x["tool"]: {"seconds": x["seconds"], "times_xrMHW": x["seconds"] / base["seconds"]}
                for x in r}

    tag = args.phase
    stem = f"bench_v2_{tag}_{prov['host']}"
    for name, r in by_phase.items():
        if r:
            pd.DataFrame(r).to_csv(outdir / f"bench_v2_{name}_{prov['host']}.csv", index=False)
    pd.DataFrame(rows).to_csv(outdir / f"{stem}.csv", index=False)
    summary["provenance"] = prov
    summary["notes"] = notes
    summary["chunk_side_used"] = best_chunk
    (outdir / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n" + "=" * 74)
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"},
                     indent=2, default=float))
    print(f"\n  rows    -> {outdir / (stem + '.csv')}")
    print(f"  summary -> {outdir / (stem + '.json')}")
    if notes:
        print("\n  notes:")
        for nte in notes:
            print(f"    - {nte}")


if __name__ == "__main__":
    main()
