"""
xrMHW benchmark, third revision (v3_3)
======================================

A copy of benchmark_xrMHW_v3_2.py, the script behind the 22-Sep-2026 draft of the
paper (results in bench_v2/ and bench_v2_proc/). That script and its results are
kept unchanged so the draft's numbers stay reproducible. This revision changes how
the out-of-core passes are timed, how xmhw is run in the reference phase, where the
temporary output goes, and what is recorded. Detection itself is untouched.

Changes from v3_2
-----------------
* Three-pass timing (see "Timing decomposition") in the sweep, scheduler and ceiling
  phases. It replaces the two-pass cold/warm subtraction, which gave a negative write
  time with worker processes.
* Reference phase: xmhw is timed with Dask's synchronous scheduler (one thread) and
  with its default threaded scheduler (--xmhw-scheduler), so the comparison with
  single-core xrMHW is like for like. The comparison can be repeated at several
  subset sizes (--ref-sides), and the number of events per cell is compared between
  the three tools.
* --scratch puts the temporary output, which is deleted after each measurement, on a
  different disk from the CSV/JSON results.
* marineHeatWaves is imported from the vendored copy in ../tests when it is not
  installed, and the file actually used is recorded.
* Results are named bench_v3_<phase>_<host>.csv and bench_v3_<tag>_<host>.json.
* Provenance adds the xmhw, zarr, netCDF4, h5py, pandas and distributed versions,
  the absolute paths and drives of the input, results and scratch folders, and a
  dask_scheduler column on every row (threads, processes, synchronous, or none for
  in-memory NumPy runs).

Phases (choose with --phase, default "all"; each writes its own CSV)
-------------------------------------------------------------------
scaling    In-memory ladder: throughput and the peak-RSS-to-raw amplification
           that sets the memory ceiling.
sweep      Chunk-size sweep on one representative 4-D subset, with detection
           time and write time measured separately.
scheduler  Threads versus worker processes. The kernel is a per-cell Python
           loop with numpy sections that release the GIL, so threads
           parallelise it only partly; worker processes sidestep the GIL.
           Give it at least as many chunks as workers.
ceiling    A domain too large for the in-memory path on this machine,
           processed out of core.
reference  xrMHW, marineHeatWaves (Oliver) and xmhw (coecms) on the SAME
           in-memory surface subset. An early run on 16 cells gave a 236x
           figure for xmhw that is not defensible; use at least 256 cells.
           xmhw's grid path breaks on some xarray versions; when that happens
           the script falls back to the same per-cell kernel, reusing the grid
           threshold, and records which path was timed.

Timing decomposition
--------------------
The sweep, scheduler and ceiling phases run three passes over the same chunked
input. With worker processes the cluster is created once and reused for all three.

  pass 0  cold cache   detection only: a reduction over every output, nothing is
                       written                                  -> detect_cold_s
  pass 1  warm cache   detection only, again                    -> detect_warm_s
  pass 2  warm cache   detection and write                      -> total_warm_s

write_s = total_warm_s - detect_warm_s compares two warm passes, so the file cache
no longer leaks into the write time. It is stored unclamped, so a negative value
stays visible. write_resolved = write_s > 2% of total_warm_s.
end_to_end_est_s = detect_cold_s + max(write_s, 0) estimates a single run that reads
the input from disk and writes the result.

detect_s keeps its v3_2 meaning (the cold detection pass) and total_s keeps its
v3_2 meaning (the warm detection-plus-write pass), so old and new CSVs can be read
by the same code. write_s does NOT keep its meaning: in v3_2 it was the cold/warm
difference.

"Cold" means the first read in this invocation. If the input is already in the
operating system's file cache (for instance because another phase read it shortly
before), pass 0 is not really cold, and cold_over_warm = detect_cold_s /
detect_warm_s close to 1 shows it. Disconnecting and reconnecting the drive that
holds the input, or rebooting, empties the cache.

peak_rss_bytes is the peak over all three passes (parent plus worker processes,
relative to the resident set when the phase started); peak_rss_pass2_bytes is the
peak during the write pass, relative to the same baseline.

Writing (--write-format)
------------------------
With worker processes, NetCDF goes through a path in which every chunk reopens the
file in append mode under a distributed lock; on the full domain one of those
reopenings failed with "Unknown file format" after 5 GB had been written. Zarr
writes one object per chunk, needs no cross-process lock, and compresses by
default, which matters because xrMHW's event fields are NaN outside event days. A
failed write does not discard the detection times.

Usage
-----
    python benchmark_xrMHW_v3_3.py --input <GLORYS.nc> --var thetao \\
        --clim 1993 2022 --phase ceiling --write-format zarr \\
        --outdir bench_v3 --scratch <folder on a disk with space>

The free-space check runs on --scratch (default: --outdir).
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
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
PREFIX = "bench_v3"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _dist_version(name):
    import importlib.metadata as md
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _where(path):
    p = Path(path).resolve()
    return str(p), (p.drive or p.anchor)


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
        "script": Path(__file__).name,
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
        "distributed": _dist_version("distributed"),
        "pandas": pd.__version__,
        "xmhw": _dist_version("xmhw"),
        "zarr": _dist_version("zarr"),
        "netCDF4": _dist_version("netCDF4"),
        "h5py": _dist_version("h5py"),
        "psutil": psutil.__version__,
        "threads_pinned_to": os.environ.get("OMP_NUM_THREADS"),
    }


def import_marineheatwaves():
    """Import Oliver's marineHeatWaves; fall back to the vendored copy in ../tests."""
    try:
        import marineHeatWaves as ref
    except ImportError:
        tests_dir = Path(__file__).resolve().parent.parent / "tests"
        if not (tests_dir / "marineHeatWaves.py").exists():
            raise
        sys.path.append(str(tests_dir))
        import marineHeatWaves as ref
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    return ref


# ---------------------------------------------------------------------------
# Peak RSS, including child processes (needed for the process scheduler)
# ---------------------------------------------------------------------------

class PeakRSS:
    """Sample the resident set of this process and its children.

    base: if given, the delta is taken against this value instead of the
    resident set sampled on entry, so a nested meter can share its parent's
    baseline.
    """

    def __init__(self, interval=0.05, include_children=True, base=None):
        self.interval = interval
        self.include_children = include_children
        self._base = base
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
        now = self._sample()
        self.base = now if self._base is None else self._base
        self.peak = now
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


def check_space(folder, needed_bytes, label):
    """Refuse to start a phase that would fill the disk halfway through.

    Every output variable is written at the same resolution as the input, so a
    run with all eight of them writes about eight times the raw array. The
    temporary output is deleted after each configuration, but it exists in full
    while the configuration runs.
    """
    free = shutil.disk_usage(folder).free
    print(f"  escritura temporal maxima estimada: {needed_bytes / 1024**3:.1f} GiB, "
          f"libre en {folder}: {free / 1024**3:.1f} GiB")
    if free < 1.2 * needed_bytes:
        sys.exit(f"Espacio insuficiente para la fase {label}: hace falta "
                 f"~{1.2 * needed_bytes / 1024**3:.1f} GiB libres en {folder} y hay "
                 f"{free / 1024**3:.1f} GiB. Usa --scratch en otro disco, reduce el "
                 f"tamano del subconjunto, o limita las salidas con --output-vars.")


def n_chunks(ds_sub, spec):
    n = 1
    for d, s in ds_sub.sizes.items():
        c = spec.get(d, s)
        c = s if c in (-1, None) else c
        n *= int(np.ceil(s / c))
    return n


class WriteFailed(RuntimeError):
    """The detection passes finished but the write did not. Carries the
    detection times so a failed write does not throw away valid measurements."""

    def __init__(self, t_cold, t_warm, n_vars, error):
        super().__init__(error)
        self.t_cold, self.t_warm, self.n_vars, self.error = t_cold, t_warm, n_vars, error


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


def time_three_pass(ds_chunked, var, clim, tmp, output_vars=None, fmt="netcdf",
                    rss_base=None):
    """Pass 0 (cold) and pass 1 (warm): detection only, forced by a reduction.
    Pass 2 (warm): detection and write. See "Timing decomposition".

    fmt: "netcdf", "zarr", or "none" (no pass 2; total_warm_s = detect_warm_s).
    rss_base: baseline for peak_rss_pass2_bytes, normally the enclosing meter's.
    Raises WriteFailed, carrying both detection times, if the write fails.
    """
    import dask

    def detect_only():
        out = xrMHW.xrMHW_func(ds_chunked, var, clim, output_vars=output_vars)
        t0 = time.perf_counter()
        dask.compute(*[v.mean() for v in out.data_vars.values()])
        return time.perf_counter() - t0, len(out.data_vars)

    t_cold, n_vars = detect_only()          # pass 0
    t_warm, _ = detect_only()               # pass 1
    res = dict(detect_cold_s=t_cold, detect_warm_s=t_warm, cold_over_warm=t_cold / t_warm,
               n_output_vars=n_vars)

    if fmt == "none":
        res.update(total_warm_s=t_warm, write_s=0.0, write_resolved=False,
                   end_to_end_est_s=t_cold, written_bytes=0, peak_rss_pass2_bytes=None)
        return res

    out = xrMHW.xrMHW_func(ds_chunked, var, clim, output_vars=output_vars)
    target = tmp.with_suffix(".zarr") if fmt == "zarr" else tmp
    _remove(target)
    try:
        with PeakRSS(base=rss_base) as rss2:
            t0 = time.perf_counter()
            if fmt == "zarr":
                # Encodings inherited from the NetCDF source (chunksizes, zlib, ...)
                # are not zarr encodings; let xarray choose fresh ones.
                for v in out.variables.values():
                    v.encoding = {}
                out.to_zarr(target, mode="w")
            else:
                out.to_netcdf(target)
            t_total = time.perf_counter() - t0  # pass 2
        written = _size(target)
    except Exception as exc:                                   # noqa: BLE001
        raise WriteFailed(t_cold, t_warm, n_vars, f"{type(exc).__name__}: {exc}") from exc
    finally:
        _remove(target)
    t_write = t_total - t_warm
    res.update(total_warm_s=t_total, write_s=t_write,
               write_resolved=bool(t_write > 0.02 * t_total),
               end_to_end_est_s=t_cold + max(t_write, 0.0), written_bytes=written,
               peak_rss_pass2_bytes=rss2.delta)
    return res


def timing_columns(t):
    """Row fields from a time_three_pass result, with the v3_2 column names kept."""
    return dict(detect_s=t["detect_cold_s"], detect_cold_s=t["detect_cold_s"],
                detect_warm_s=t["detect_warm_s"], cold_over_warm=t["cold_over_warm"],
                total_s=t["total_warm_s"], total_warm_s=t["total_warm_s"],
                write_s=t["write_s"], write_resolved=t["write_resolved"],
                end_to_end_est_s=t["end_to_end_est_s"], written_bytes=t["written_bytes"],
                peak_rss_pass2_bytes=t["peak_rss_pass2_bytes"])


def failed_columns(wf):
    return dict(detect_s=wf.t_cold, detect_cold_s=wf.t_cold, detect_warm_s=wf.t_warm,
                cold_over_warm=wf.t_cold / wf.t_warm, total_s=None, total_warm_s=None,
                write_s=None, write_resolved=False, end_to_end_est_s=None, written_bytes=0,
                peak_rss_pass2_bytes=None)


# ---------------------------------------------------------------------------
# Phase: in-memory scaling
# ---------------------------------------------------------------------------

def phase_scaling(ds, var, clim, budget_frac):
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
        rows.append(dict(phase="scaling", label=label, dask_scheduler="none",
                         cells=cells, timesteps=nt, raw_bytes=raw, seconds=elapsed,
                         cell_steps_per_s=cells * nt / elapsed,
                         peak_rss_bytes=rss.delta, peak_over_raw=rss.delta / raw))
        print(f"  {label:12s} {raw / BYTES_PER_MIB:8.1f} MiB  {elapsed:7.1f} s  "
              f"{cells * nt / elapsed / 1e3:7.0f} k cell-steps/s  peak {rss.delta / raw:5.2f}x")
    return rows


# ---------------------------------------------------------------------------
# Phase: chunk sweep
# ---------------------------------------------------------------------------

def phase_sweep(ds, var, clim, scratch, depth, side, sides, output_vars=None,
                threads=1, amplification=7.26, budget_frac=0.75, fmt="netcdf"):
    sub = subset(ds, var, depth, side)
    cells, nt, raw = shape_of(sub, var)
    tmp = scratch / "_sweep.nc"
    budget = budget_frac * psutil.virtual_memory().total
    rows = []
    print("\n" + "=" * 74)
    print(f" PHASE sweep — chunk size on d{depth}x{side}x{side} "
          f"({raw / 1024**3:.2f} GiB raw, {cells} cells)")
    print("=" * 74)
    if fmt != "none":
        check_space(scratch, raw * (len(output_vars) if output_vars else 8), "sweep")
    print(f"  presupuesto de memoria: {budget / 1024**3:.0f} GiB "
          f"({budget_frac:.0%} de la RAM), {threads} hilos concurrentes")
    print(f"  {'chunk side':>10} {'chunks':>7} {'MiB/chunk':>10} {'proy GiB':>9} {'cold':>8} "
          f"{'warm':>8} {'write':>8} {'e2e':>8} {'peak GiB':>9}")
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
        # threaded scheduler is the thread count.
        projected = min(threads, nch) * per_chunk * amplification
        base = dict(phase="sweep", label=f"d{depth}x{side}x{side}", dask_scheduler="threads",
                    chunk_side=cs, n_chunks=nch, mib_per_chunk=per_chunk / BYTES_PER_MIB,
                    projected_peak_bytes=projected)
        if projected > budget:
            print(f"  {cs:>10} {nch:>7} {per_chunk / BYTES_PER_MIB:>10.1f} "
                  f"{projected / 1024**3:>9.1f}   omitido: proyeccion sobre el presupuesto")
            rows.append(dict(base, skipped_over_budget=True))
            continue
        chunked = sub.chunk(spec)
        with PeakRSS() as rss:
            t = time_three_pass(chunked, var, clim, tmp, output_vars, fmt, rss_base=rss.base)
        rows.append(dict(base, skipped_over_budget=False, write_format=fmt,
                         cells=cells, timesteps=nt, raw_bytes=raw,
                         **timing_columns(t), n_output_vars=t["n_output_vars"],
                         peak_rss_bytes=rss.delta,
                         cell_steps_per_s=cells * nt / t["end_to_end_est_s"]))
        print(f"  {cs:>10} {nch:>7} {per_chunk / BYTES_PER_MIB:>10.1f} "
              f"{projected / 1024**3:>9.1f} {t['detect_cold_s']:>8.1f} {t['detect_warm_s']:>8.1f} "
              f"{t['write_s']:>8.1f} {t['end_to_end_est_s']:>8.1f} {rss.delta / 1024**3:>9.2f}")
    measured = [r for r in rows if not r["skipped_over_budget"]]
    if measured:
        best = min(measured, key=lambda r: r["end_to_end_est_s"])
        print(f"\n  optimum at chunk side {best['chunk_side']} "
              f"({best['mib_per_chunk']:.0f} MiB/chunk, {best['end_to_end_est_s']:.1f} s end to end); "
              f"worst/best = {max(r['end_to_end_est_s'] for r in measured) / best['end_to_end_est_s']:.1f}x")
        err = [abs(r["peak_rss_bytes"] - r["projected_peak_bytes"]) / r["peak_rss_bytes"]
               for r in measured if r["peak_rss_bytes"] > 0]
        if err:
            print(f"  error medio del modelo de memoria "
                  f"(hilos x chunk x {amplification:.2f}): {100 * sum(err) / len(err):.0f}%")
    return rows


# ---------------------------------------------------------------------------
# Phase: scheduler
# ---------------------------------------------------------------------------

def phase_scheduler(ds, var, clim, scratch, depth, side, chunk_side, worker_counts,
                    output_vars=None, fmt="netcdf"):
    sub = subset(ds, var, depth, side)
    cells, nt, raw = shape_of(sub, var)
    spec = chunk_spec(sub, chunk_side)
    chunked = sub.chunk(spec)
    tmp = scratch / "_sched.nc"
    rows = []
    print("\n" + "=" * 74)
    print(f" PHASE scheduler — threads vs processes, d{depth}x{side}x{side}, "
          f"chunk side {chunk_side} ({n_chunks(sub, spec)} chunks), escritura: {fmt}")
    print("=" * 74)
    if fmt != "none":
        check_space(scratch, raw * (len(output_vars) if output_vars else 8), "scheduler")

    def run(name, sched, workers):
        error = None
        with PeakRSS() as rss:
            try:
                t = time_three_pass(chunked, var, clim, tmp, output_vars, fmt, rss_base=rss.base)
                cols = timing_columns(t)
            except WriteFailed as wf:
                cols, error = failed_columns(wf), wf.error
        row = dict(phase="scheduler", scheduler=name, dask_scheduler=sched, workers=workers,
                   write_format=fmt, write_error=error, chunk_side=chunk_side,
                   n_chunks=n_chunks(sub, spec), cells=cells, timesteps=nt, raw_bytes=raw,
                   **cols, peak_rss_bytes=rss.delta,
                   detect_cell_steps_per_s=cells * nt / cols["detect_cold_s"],
                   cell_steps_per_s=(cells * nt / cols["end_to_end_est_s"])
                   if cols["end_to_end_est_s"] else None)
        tail = (f"write {cols['write_s']:7.1f}  e2e {cols['end_to_end_est_s']:7.1f}"
                if cols["end_to_end_est_s"] is not None else "write FALLIDA (deteccion conservada)")
        print(f"  {name + (f' x{workers}' if workers else ''):<26} cold {cols['detect_cold_s']:7.1f}  "
              f"warm {cols['detect_warm_s']:7.1f}  {tail}  peak {rss.delta / 1024**3:5.2f} GiB  "
              f"({cells * nt / cols['detect_cold_s'] / 1e6:5.2f} M cell-steps/s)")
        if error:
            print(f"      error de escritura: {error[:200]}")
        return row

    rows.append(run("threads (default)", "threads", None))

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
            rows.append(run("processes", "processes", nw))
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

def phase_ceiling(ds, var, clim, scratch, amplification, chunk_side, side,
                  output_vars=("severity",), workers=0, fmt="netcdf"):
    total_ram = psutil.virtual_memory().total
    depth = ds.sizes.get("depth", 1)
    side = side or max(ds.sizes.get("latitude", 1), ds.sizes.get("longitude", 1))
    sub = subset(ds, var, depth, side)
    cells, nt, raw = shape_of(sub, var)
    projected = amplification * raw
    spec = chunk_spec(sub, chunk_side)
    tmp = scratch / "_ceiling.nc"
    output_vars = list(output_vars) if output_vars else None

    print("\n" + "=" * 74)
    print(" PHASE ceiling — a domain the in-memory path cannot hold")
    print("=" * 74)
    label = "d" + "x".join(str(sub.sizes[d]) for d in ("depth", "latitude", "longitude")
                           if d in sub.sizes)
    print(f"  domain {label}: {cells} cells, {raw / 1024**3:.2f} GiB raw")
    print(f"  salidas escritas: {', '.join(output_vars) if output_vars else 'todas (8)'}")
    print(f"  in-memory would need ~{projected / 1024**3:.0f} GiB "
          f"({amplification:.2f}x raw) against {total_ram / 1024**3:.0f} GiB installed"
          f"  -> {'DOES NOT FIT' if projected > total_ram else 'fits'}")
    print(f"  in-memory ceiling on this machine: "
          f"{total_ram / amplification / 1024**3:.1f} GiB of input")

    if fmt != "none":
        check_space(scratch, raw * (len(output_vars) if output_vars else 8), "ceiling")
    chunked = sub.chunk(spec)

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
    print("  tres pasadas: 0 deteccion (frio), 1 deteccion (caliente), 2 deteccion + escritura")
    error = None
    try:
        with PeakRSS() as rss:
            try:
                t = time_three_pass(chunked, var, clim, tmp, output_vars, fmt, rss_base=rss.base)
                cols, nvar = timing_columns(t), t["n_output_vars"]
            except WriteFailed as wf:
                cols, nvar, error = failed_columns(wf), wf.n_vars, wf.error
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

    print(f"  deteccion en frio: {cols['detect_cold_s']:.1f} s = {cols['detect_cold_s'] / 60:.1f} min, "
          f"{cells * nt / cols['detect_cold_s'] / 1e6:.2f} M cell-steps/s; "
          f"en caliente: {cols['detect_warm_s']:.1f} s (frio/caliente {cols['cold_over_warm']:.2f}); "
          f"pico {rss.delta / 1024**3:.2f} GiB")
    if error:
        print(f"  ESCRITURA FALLIDA ({fmt}); la deteccion se conserva. Error: {error[:300]}")
    elif fmt != "none":
        print(f"  escritura: {cols['write_s']:.1f} s "
              f"({'resuelta' if cols['write_resolved'] else 'NO resuelta'}), "
              f"extremo a extremo estimado {cols['end_to_end_est_s'] / 60:.1f} min, "
              f"escribio {cols['written_bytes'] / 1024**3:.2f} GiB")
    return [dict(phase="ceiling", label=label,
                 dask_scheduler="processes" if workers else "threads",
                 workers=workers or None, write_format=fmt, write_error=error,
                 cells=cells, timesteps=nt,
                 raw_bytes=raw, projected_in_memory_bytes=projected,
                 fits_in_memory=bool(projected <= total_ram), chunk_side=chunk_side,
                 n_chunks=n_chunks(sub, spec), n_output_vars=nvar, **cols,
                 peak_rss_bytes=rss.delta, peak_over_raw=rss.delta / raw,
                 detect_cell_steps_per_s=cells * nt / cols["detect_cold_s"],
                 cell_steps_per_s=(cells * nt / cols["end_to_end_est_s"])
                 if cols["end_to_end_est_s"] else None)]


# ---------------------------------------------------------------------------
# Phase: reference implementations, same subset for all three
# ---------------------------------------------------------------------------

def _drop_depth(da):
    return da.isel(depth=0, drop=True) if "depth" in da.dims else da


def _runs_along_time(mask):
    """Number of runs of True along time, i.e. events per cell."""
    prev = mask.shift(time=1, fill_value=False)
    return (mask & ~prev).sum("time")


def _compare_counts(a, b):
    a, b = xr.align(a, b, join="inner")
    ok = a.notnull() & b.notnull()
    n = int(ok.sum())
    diff = abs(a - b).where(ok)
    return {"cells_compared": n,
            "mismatches": int((diff > 0).sum()),
            "max_abs_diff": float(diff.max()) if n else None,
            "events_total_xrMHW": float(a.where(ok).sum()),
            "events_total_other": float(b.where(ok).sum())}


def _run_xmhw(da, clim, scheduler, notes):
    """Time xmhw's threshold + detect under the given Dask scheduler.

    scheduler: "synchronous" (one thread) or "threads" (the global threaded
    configuration, i.e. num_workers Dask threads). Returns
    (path, threshold_s, detect_s, result_or_None).
    """
    import dask
    from xmhw.xmhw import detect as xmhw_detect, threshold as xmhw_threshold

    ctx = (dask.config.set(scheduler="synchronous") if scheduler == "synchronous"
           else contextlib.nullcontext())
    with ctx:
        t0 = time.perf_counter()
        clim_ds = xmhw_threshold(da, climatologyPeriod=list(clim)).compute()
        t_thresh = time.perf_counter() - t0

        # Preferred: the grid path, which is how xmhw is meant to be used.
        # Some xarray versions break its multi-index reassembly; fall back to the
        # same per-cell kernel one cell at a time, reusing the grid threshold.
        path, res, t0 = "grid", None, time.perf_counter()
        try:
            res = xmhw_detect(da, clim_ds.thresh, clim_ds.seas)
            res = res.compute() if hasattr(res, "compute") else res
        except Exception as exc:                               # noqa: BLE001
            notes.append(f"xmhw grid path failed under {scheduler} ({type(exc).__name__}: "
                         f"{exc}); timed the per-cell path with the grid threshold instead")
            path, res, t0 = "per-cell", None, time.perf_counter()
            hdims = [d for d in da.dims if d != "time"]
            for pos in np.ndindex(*[da.sizes[d] for d in hdims]):
                sel = dict(zip(hdims, pos))
                if bool(np.isnan(da.isel(sel).values).all()):
                    continue
                xmhw_detect(da.isel(sel), clim_ds.thresh.isel(sel), clim_ds.seas.isel(sel))
        t_det = time.perf_counter() - t0
    return path, t_thresh, t_det, res


def phase_reference(ds, var, clim, side, xmhw_scheduler, threads, prov):
    sub = subset(ds, var, 1, side)
    cells, nt, raw = shape_of(sub, var)
    loaded = sub.load()
    rows, notes, counts = [], [], {}
    print("\n" + "=" * 74)
    print(f" PHASE reference — same subset for all three: d1x{side}x{side} "
          f"({cells} cells, {raw / BYTES_PER_MIB:.1f} MiB), in memory")
    print("=" * 74)
    common = dict(phase="reference", ref_side=side, timesteps=nt)

    t0 = time.perf_counter()
    out = xrMHW.xrMHW_func(loaded, var, clim)
    _ = out["mhw_intensity"].values
    t_xr = time.perf_counter() - t0
    rows.append(dict(common, tool="xrMHW (in-memory)", dask_scheduler="none", threads=1,
                     cells=cells, seconds=t_xr, cell_steps_per_s=cells * nt / t_xr))
    print(f"  {'xrMHW, 1 core':<32} {t_xr:8.1f} s   {cells * nt / t_xr / 1e3:8.0f} k cell-steps/s")
    try:
        ocean = _drop_depth(loaded[var].notnull().any("time"))
        counts["xrMHW"] = _drop_depth(_runs_along_time(out["mhw_duration"].notnull())).where(ocean)
    except Exception as exc:                                   # noqa: BLE001
        notes.append(f"xrMHW event counts unavailable: {type(exc).__name__}: {exc}")
    del out

    try:
        ref = import_marineheatwaves()
        prov["marineHeatWaves_origin"] = str(Path(ref.__file__).resolve())
        flat = loaded[var].stack(cell=[d for d in loaded[var].dims if d != "time"])
        t_ord = np.array([d.toordinal() for d in loaded.indexes["time"].to_pydatetime()])
        n_ev = np.full(flat.sizes["cell"], np.nan)
        t0, n_ok = time.perf_counter(), 0
        for k in range(flat.sizes["cell"]):
            ts = np.asarray(flat.isel(cell=k).values, dtype=float)
            if np.isnan(ts).all():
                continue
            mhw_k, _ = ref.detect(t_ord, ts.copy(), climatologyPeriod=list(clim))
            n_ev[k] = mhw_k["n_events"]
            n_ok += 1
        t_ref = time.perf_counter() - t0
        rows.append(dict(common, tool="marineHeatWaves (cell by cell)", dask_scheduler="none",
                         threads=1, cells=n_ok, seconds=t_ref, cell_steps_per_s=n_ok * nt / t_ref))
        print(f"  {'marineHeatWaves, 1 core':<32} {t_ref:8.1f} s   {n_ok * nt / t_ref / 1e3:8.0f} "
              f"k cell-steps/s   ({t_ref / t_xr:.2f}x xrMHW, {n_ok} ocean cells)")
        try:
            counts["marineHeatWaves"] = _drop_depth(
                flat.isel(time=0, drop=True).copy(data=n_ev).unstack("cell"))
        except Exception as exc:                               # noqa: BLE001
            notes.append(f"marineHeatWaves event counts unavailable: {type(exc).__name__}: {exc}")
    except ImportError:
        notes.append("marineHeatWaves not importable (not installed and no ../tests copy)")
        print("  marineHeatWaves                  not importable — see note in the JSON")

    schedulers = ["synchronous", "threads"] if xmhw_scheduler == "both" else \
                 ["synchronous" if xmhw_scheduler == "synchronous" else "threads"]
    try:
        import xmhw  # noqa: F401
        # xmhw expects the horizontal grid only; a length-one depth dimension
        # survives `threshold` but breaks the reassembly inside `detect`.
        da = loaded[var]
        da = da.squeeze(drop=True) if "depth" in da.dims else da
        for sched in schedulers:
            n_thr = 1 if sched == "synchronous" else threads
            path, t_thresh, t_det, res = _run_xmhw(da, clim, sched, notes)
            t_xm = t_thresh + t_det
            tool = f"xmhw [{path}, {sched}]"
            rows.append(dict(common, tool=tool, dask_scheduler=sched, threads=n_thr,
                             cells=cells, seconds=t_xm, threshold_s=t_thresh, detect_s=t_det,
                             cell_steps_per_s=cells * nt / t_xm))
            name = f"xmhw, {n_thr} thread" + ("s" if n_thr > 1 else "")
            print(f"  {name:<32} {t_xm:8.1f} s   "
                  f"{cells * nt / t_xm / 1e3:8.0f} k cell-steps/s   ({t_xm / t_xr:.1f}x xrMHW; "
                  f"threshold {t_thresh:.1f} s = {100 * t_thresh / t_xm:.0f}%, detect {t_det:.1f} s, "
                  f"{path})")
            if res is not None and "xmhw" not in counts:
                try:
                    v = "duration" if "duration" in res else next(
                        n for n in res.data_vars if "events" in res[n].dims)
                    counts["xmhw"] = res[v].notnull().sum("events")
                except Exception as exc:                       # noqa: BLE001
                    notes.append(f"xmhw event counts unavailable: {type(exc).__name__}: {exc}")
    except Exception as exc:                                   # noqa: BLE001
        notes.append(f"xmhw not usable: {type(exc).__name__}: {exc}")
        print(f"  xmhw                             not usable ({type(exc).__name__}: {exc})")

    agreement = {}
    if "xrMHW" in counts:
        for other in ("marineHeatWaves", "xmhw"):
            if other in counts:
                try:
                    agreement[other] = _compare_counts(counts["xrMHW"], counts[other])
                except Exception as exc:                       # noqa: BLE001
                    notes.append(f"event-count comparison with {other} failed: "
                                 f"{type(exc).__name__}: {exc}")
    for other, a in agreement.items():
        print(f"  eventos por celda, xrMHW vs {other}: {a['mismatches']} de "
              f"{a['cells_compared']} celdas difieren (max |diff| {a['max_abs_diff']})")
        if a["mismatches"]:
            notes.append(f"side {side}: event counts differ from {other} in {a['mismatches']} "
                         f"of {a['cells_compared']} cells (max abs diff {a['max_abs_diff']})")
    return rows, notes, agreement


def reference_summary(rows, agreement):
    base = next((x for x in rows if x["tool"].startswith("xrMHW")), None)
    if base is None:
        return {}
    out = {}
    for x in rows:
        d = {"seconds": x["seconds"], "times_xrMHW": x["seconds"] / base["seconds"],
             "cells": x["cells"], "threads": x.get("threads"),
             "dask_scheduler": x.get("dask_scheduler")}
        if x.get("threshold_s") is not None and not pd.isna(x.get("threshold_s")):
            d["threshold_s"] = x["threshold_s"]
            d["detect_s"] = x["detect_s"]
            d["threshold_share_percent"] = 100 * x["threshold_s"] / x["seconds"]
        out[x["tool"]] = d
    out["event_count_agreement"] = agreement
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--var", default="thetao")
    ap.add_argument("--clim", nargs=2, type=int, default=[1993, 2022], metavar=("A", "B"))
    ap.add_argument("--outdir", default=".", help="where the CSV and JSON results go")
    ap.add_argument("--scratch", default=None,
                    help="folder for the temporary output written and deleted by the sweep, "
                         "scheduler and ceiling phases (default: --outdir). The free-space "
                         "check runs here.")
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
                    help="outputs written in the ceiling phase (default: severity only); "
                         "pass 'all' to write all eight")
    ap.add_argument("--write-format", choices=["netcdf", "zarr", "none"], default="netcdf",
                    help="how the sweep, scheduler and ceiling phases write their output. "
                         "'none' times detection only. With worker processes use zarr.")
    ap.add_argument("--ceiling-workers", type=int, default=0,
                    help="run the ceiling phase on this many worker processes; 0 (default) "
                         "uses the threaded scheduler")
    ap.add_argument("--ceiling-side", type=int, default=0,
                    help="horizontal side for the ceiling demonstration; 0 uses the whole "
                         "horizontal domain")
    ap.add_argument("--ref-side", type=int, default=16,
                    help="horizontal side of the surface subset shared by all three "
                         "implementations (16 -> 256 cells)")
    ap.add_argument("--ref-sides", type=int, nargs="+", default=None,
                    help="repeat the reference comparison at several sides, e.g. 16 32; "
                         "overrides --ref-side")
    ap.add_argument("--xmhw-scheduler", choices=["default", "synchronous", "both"],
                    default="both",
                    help="Dask scheduler for xmhw in the reference phase: 'synchronous' is "
                         "one thread, like xrMHW and marineHeatWaves there; 'default' is the "
                         "threaded scheduler with --threads workers, which is what v3_2 timed")
    ap.add_argument("--threads", type=int, default=None,
                    help="threads for Dask's default scheduler; defaults to the physical "
                         "core count")
    ap.add_argument("--amplification", type=float, default=None,
                    help="peak-RSS-to-raw ratio used by the sweep's memory model and by the "
                         "ceiling projection when the scaling phase is not run in the same "
                         "invocation (default: the measured value, else 7.26)")
    ap.add_argument("--budget-frac", type=float, default=0.6)
    ap.add_argument("--sweep-budget-frac", type=float, default=0.75)
    ap.add_argument("--min-fit-mib", type=float, default=64.0)
    ap.add_argument("--allow-path-import", action="store_true")
    args = ap.parse_args()

    prov = provenance(args.allow_path_import)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch) if args.scratch else outdir
    scratch.mkdir(parents=True, exist_ok=True)
    prov["input_path"], prov["input_drive"] = _where(args.input)
    prov["outdir"], prov["outdir_drive"] = _where(outdir)
    prov["scratch"], prov["scratch_drive"] = _where(scratch)
    clim = tuple(args.clim)
    ds = xr.open_dataset(args.input)

    import dask
    threads = args.threads or psutil.cpu_count(logical=False) or 4
    dask.config.set(num_workers=threads)
    prov["dask_threads"] = threads

    print("=" * 74)
    print(f" xrMHW {prov['xrMHW_version']} benchmark v3_3 — {prov['host']}, {prov['cpu'][:34]}")
    print(f" {prov['physical_cores']} physical cores, {prov['total_ram_gib']} GiB RAM, "
          f"{prov['threads_pinned_to']} BLAS thread pinned, {threads} Dask threads")
    print(f" imported from {prov['xrMHW_origin']} ({prov['xrMHW_install_mode']})")
    print(f" input {prov['input_path']}")
    print(f" results -> {prov['outdir']}   temporary output -> {prov['scratch']}")
    print("=" * 74)

    want = args.phase
    rows, notes, summary, by_phase = [], [], {}, {}

    def keep(name, r):
        by_phase[name] = r
        rows.extend(r)

    amplification = args.amplification or 7.26   # fallback if scaling is not run here
    if want in ("all", "scaling"):
        r = phase_scaling(ds, args.var, clim, args.budget_frac)
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
        r = phase_sweep(ds, args.var, clim, scratch,
                        args.sweep_depth, args.sweep_side, args.sweep_chunks,
                        args.output_vars, threads, amplification, args.sweep_budget_frac,
                        args.write_format)
        keep("sweep", r)
        r = [x for x in r if not x["skipped_over_budget"]]
        if r:
            b = min(r, key=lambda x: x["end_to_end_est_s"])
            best_chunk = args.chunk_side or b["chunk_side"]
            summary["sweep"] = {"best_chunk_side": b["chunk_side"],
                                "best_mib_per_chunk": b["mib_per_chunk"],
                                "best_end_to_end_est_s": b["end_to_end_est_s"],
                                "worst_over_best": max(x["end_to_end_est_s"] for x in r)
                                / b["end_to_end_est_s"],
                                "write_fraction_at_best": max(b["write_s"], 0.0)
                                / b["end_to_end_est_s"]}

    if want in ("all", "scheduler"):
        r = phase_scheduler(ds, args.var, clim, scratch,
                            args.sweep_depth, args.sweep_side, best_chunk, args.workers,
                            args.output_vars, args.write_format)
        keep("scheduler", r)
        thr = next((x for x in r if x["workers"] is None), None)
        proc = [x for x in r if x["workers"] is not None]
        if thr and proc:
            fd = min(proc, key=lambda x: x["detect_cold_s"])
            summary["scheduler_detection"] = {
                "threads_detect_cold_s": thr["detect_cold_s"],
                "best_processes_workers": fd["workers"],
                "best_processes_detect_cold_s": fd["detect_cold_s"],
                "process_detect_speedup_over_threads": thr["detect_cold_s"] / fd["detect_cold_s"],
            }
        thr = thr if (thr and thr["end_to_end_est_s"]) else None
        proc = [x for x in proc if x["end_to_end_est_s"]]
        if thr and proc:
            fastest = min(proc, key=lambda x: x["end_to_end_est_s"])
            summary["scheduler"] = {
                "threads_end_to_end_est_s": thr["end_to_end_est_s"],
                "best_processes_workers": fastest["workers"],
                "best_processes_end_to_end_est_s": fastest["end_to_end_est_s"],
                "process_speedup_over_threads":
                    thr["end_to_end_est_s"] / fastest["end_to_end_est_s"],
            }

    if want in ("all", "ceiling"):
        cov = args.ceiling_output_vars
        cov = None if (cov and cov[0].lower() == "all") else cov
        r = phase_ceiling(ds, args.var, clim, scratch, amplification, best_chunk,
                          args.ceiling_side, cov, args.ceiling_workers, args.write_format)
        keep("ceiling", r)
        c = r[0]
        summary["ceiling"] = {k: c[k] for k in (
            "dask_scheduler", "workers", "write_format", "chunk_side", "n_chunks",
            "detect_cold_s", "detect_warm_s", "cold_over_warm", "write_s", "write_resolved",
            "total_warm_s", "end_to_end_est_s", "peak_rss_bytes", "peak_rss_pass2_bytes",
            "written_bytes", "detect_cell_steps_per_s", "write_error")}

    if want in ("all", "reference"):
        sides = args.ref_sides or [args.ref_side]
        all_r = []
        summary["reference"] = {}
        for side in sides:
            r, n, agreement = phase_reference(ds, args.var, clim, side, args.xmhw_scheduler,
                                              threads, prov)
            all_r += r
            notes += n
            summary["reference"][str(side)] = reference_summary(r, agreement)
        keep("reference", all_r)

    tag = args.phase
    host = prov["host"]
    stem = f"{PREFIX}_{tag}_{host}"
    for name, r in by_phase.items():
        if r:
            pd.DataFrame(r).to_csv(outdir / f"{PREFIX}_{name}_{host}.csv", index=False)
    if tag == "all":
        pd.DataFrame(rows).to_csv(outdir / f"{stem}.csv", index=False)
    summary["provenance"] = prov
    summary["notes"] = notes + [
        "detect_cold_s is cold only if the input was not in the OS file cache when the run "
        "started; cold_over_warm close to 1 means it was already cached"]
    summary["chunk_side_used"] = best_chunk
    (outdir / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n" + "=" * 74)
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"},
                     indent=2, default=float))
    print(f"\n  rows    -> {outdir / (PREFIX + '_<phase>_' + host + '.csv')}")
    print(f"  summary -> {outdir / (stem + '.json')}")
    if notes:
        print("\n  notes:")
        for nte in notes:
            print(f"    - {nte}")


if __name__ == "__main__":
    main()
