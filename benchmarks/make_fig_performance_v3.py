"""Figure 1 of the xrMHW paper, v3: memory (A) and detection throughput (B).

A copy of make_fig_performance.py that reads each phase from its own CSV, so phases
re-run with benchmark_xrMHW_v3_3.py (bench_v3*/) can be combined with phases kept
from benchmark_xrMHW_v3_2.py (bench_v2*/). It reads either CSV format.

Run from the benchmarks folder:

    python make_fig_performance_v3.py ^
        --scaling   bench_v2/bench_v2_scaling_Gabo.csv ^
        --sweep     bench_v2/bench_v2_sweep_Gabo.csv ^
        --sched     bench_v2_proc/bench_v2_scheduler_Gabo.csv ^
        --reference bench_v3/bench_v3_reference_Gabo.csv ^
        --ceil-thr  bench_v3/bench_v3_ceiling_Gabo.csv ^
        --ceil-proc bench_v3_proc/bench_v3_ceiling_Gabo.csv ^
        --labels memory

--labels memory  the two full-domain markers in panel A are labelled with their
                 peak memory and no times;
--labels time    both are labelled with the same two stages: detection (the cold
                 detection pass, detect_cold_s) and end to end (end_to_end_est_s).

Panel B uses the cold detection pass for the full-domain throughput and shows xmhw
on one thread and on its default threaded scheduler, or only on one thread with
--xmhw-sync-only. Values are printed with three significant figures.

Writes <out>.png (300 dpi) and <out>.pdf; --out defaults to fig_performance_v3_<labels>.
It never writes fig_performance.png, the figure of the 22-Sep-2026 draft.
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

GiB = 2 ** 30
# categorical slots 1-3 (validated all-pairs on white); references in muted gray
C_MEM, C_THR, C_PROC, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
M_MEM, M_THR, M_PROC = "o", "s", "^"
RAM_GIB = 63.7


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelcolor=INK2, width=0.8, length=3)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def gib_fmt(v, _):
    return f"{v:g}"


def sig3(v):
    """Three significant figures, trailing zeros kept:
    0.0216 -> '0.0216', 0.1297 -> '0.130', 113.4 -> '113', 2.441 -> '2.44'."""
    return f"{v:#.3g}".rstrip(".")


def minutes(seconds):
    m = seconds / 60
    return f"{m:.0f}" if m >= 10 else f"{m:.1f}"


def gib_label(nbytes):
    g = nbytes / GiB
    return f"{g:.0f}" if g >= 10 else f"{g:.1f}"


def col(row, *names):
    """First of `names` present and not NaN in a CSV row (v3 names first, v2 fallback)."""
    for n in names:
        if n in row and pd.notna(row[n]):
            return float(row[n])
    return np.nan


def dims_text(label):
    m = re.fullmatch(r"d(\d+)x(\d+)x(\d+)", str(label))
    return " × ".join(m.groups()) if m else str(label)


def reference_rows(ref, side, sync_only):
    """Panel B rows for the reference group, from a v2 or v3 reference CSV."""
    if "ref_side" in ref.columns and ref.ref_side.notna().any():
        sides = sorted(int(s) for s in ref.ref_side.dropna().unique())
        ref = ref[ref.ref_side == (side if side in sides else sides[0])]
    xr_row = ref[ref.tool.str.startswith("xrMHW")].iloc[0]
    xr_one = xr_row.cell_steps_per_s
    rows = [(f"Same {int(xr_row.cells)} surface cells", None, None, None, None)]

    xm = ref[ref.tool.str.startswith("xmhw")].copy()
    if "dask_scheduler" in xm.columns and xm.dask_scheduler.notna().any():
        # threaded run first (top), then the one-thread run
        xm["order"] = xm.dask_scheduler.map({"threads": 0, "synchronous": 1}).fillna(2)
        xm = xm.sort_values("order")
        if sync_only:
            xm = xm[xm.dask_scheduler == "synchronous"]
        for _, x in xm.iterrows():
            n = int(x.threads) if "threads" in x and pd.notna(x.threads) else None
            name = "xmhw, 1 thread" if x.dask_scheduler == "synchronous" else \
                   f"xmhw, {n} threads" if n else "xmhw, threads"
            rows.append((name, x.cell_steps_per_s, C_REF, "D",
                         f"{sig3(xr_one / x.cell_steps_per_s)}× slower"))
    else:
        for _, x in xm.iterrows():
            rows.append(("xmhw", x.cell_steps_per_s, C_REF, "D",
                         f"{sig3(xr_one / x.cell_steps_per_s)}× slower"))

    mhw = ref[ref.tool.str.startswith("marineHeatWaves")]
    if len(mhw):
        m = mhw.iloc[0]
        rows.append(("marineHeatWaves, cell by cell", m.cell_steps_per_s, C_REF, "D",
                     f"{sig3(xr_one / m.cell_steps_per_s)}× slower"))
    rows.append(("xrMHW, in memory, 1 core", xr_one, C_MEM, M_MEM, None))
    return rows, xr_one


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaling", default="bench_v2/bench_v2_scaling_Gabo.csv")
    ap.add_argument("--sweep", default="bench_v2/bench_v2_sweep_Gabo.csv")
    ap.add_argument("--sched", default="bench_v2_proc/bench_v2_scheduler_Gabo.csv")
    ap.add_argument("--reference", default="bench_v3/bench_v3_reference_Gabo.csv")
    ap.add_argument("--ceil-thr", default="bench_v3/bench_v3_ceiling_Gabo.csv")
    ap.add_argument("--ceil-proc", default="bench_v3_proc/bench_v3_ceiling_Gabo.csv")
    ap.add_argument("--labels", choices=["memory", "time"], default="memory")
    ap.add_argument("--ref-side", type=int, default=16,
                    help="which reference subset to plot when the CSV holds several")
    ap.add_argument("--xmhw-sync-only", action="store_true",
                    help="show only xmhw on one thread in panel B")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"fig_performance_v3_{a.labels}"
    if out.replace("\\", "/").split("/")[-1] == "fig_performance":
        raise SystemExit("refusing to overwrite fig_performance.*, the 22-Sep-2026 figure")

    scal = pd.read_csv(a.scaling)
    sweep = pd.read_csv(a.sweep)
    if "skipped_over_budget" in sweep:
        sweep = sweep[~sweep.skipped_over_budget.astype(bool)]
    ref = pd.read_csv(a.reference)
    ceil_thr = pd.read_csv(a.ceil_thr).iloc[0]
    sched = pd.read_csv(a.sched)
    ceil_proc = pd.read_csv(a.ceil_proc).iloc[0]

    # in-memory fit on subsets >= 64 MiB, as in the benchmark
    fit = scal[scal.raw_bytes >= 64 * 2 ** 20]
    slope, icpt = np.polyfit(fit.raw_bytes, fit.peak_rss_bytes, 1)
    r2 = np.corrcoef(fit.raw_bytes, fit.peak_rss_bytes)[0, 1] ** 2
    ceiling = (RAM_GIB * GiB - icpt) / slope / GiB
    full_raw = ceil_proc.raw_bytes / GiB
    full_proj = (slope * ceil_proc.raw_bytes + icpt) / GiB

    thr_detect = col(ceil_thr, "detect_cold_s", "detect_s")
    proc_detect = col(ceil_proc, "detect_cold_s", "detect_s")
    thr_e2e = col(ceil_thr, "end_to_end_est_s")
    proc_e2e = col(ceil_proc, "end_to_end_est_s")
    n_proc = int(col(ceil_proc, "workers")) if pd.notna(col(ceil_proc, "workers")) else 14

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig = plt.figure(figsize=(6.3, 7.9))
    axA = fig.add_axes([0.10, 0.575, 0.86, 0.39])
    axB = fig.add_axes([0.385, 0.07, 0.575, 0.39])

    # ------------------------------------------------------------------ A
    style_axes(axA)
    axA.set_xscale("log"); axA.set_yscale("log")
    xs = np.geomspace(0.05, scal.raw_bytes.max() / GiB, 50)
    axA.plot(xs, (slope * xs * GiB + icpt) / GiB, color=C_MEM, lw=1.4, zorder=2)
    xp = np.geomspace(scal.raw_bytes.max() / GiB, max(full_raw, scal.raw_bytes.max() / GiB), 50)
    axA.plot(xp, (slope * xp * GiB + icpt) / GiB, color=C_MEM, lw=1.1, ls=(0, (4, 3)), zorder=2)
    axA.axhline(RAM_GIB, color=INK2, lw=0.9, zorder=1)
    axA.text(0.0125, RAM_GIB * 1.12, "installed RAM, 64 GB", color=INK2, fontsize=8, va="bottom")

    ring = dict(edgecolors="white", linewidths=0.9, zorder=4)
    axA.scatter(scal.raw_bytes / GiB, scal.peak_rss_bytes / GiB, s=30, c=C_MEM, marker=M_MEM, **ring)
    axA.scatter([full_raw], [full_proj], s=36, facecolors="white", edgecolors=C_MEM,
                linewidths=1.3, marker=M_MEM, zorder=4)
    axA.scatter([ceiling], [RAM_GIB], s=34, facecolors="white", edgecolors=INK, linewidths=1.1,
                marker="D", zorder=5)

    thr_x = list(sweep.raw_bytes / GiB)
    thr_y = list(sweep.peak_rss_bytes / GiB)
    t48 = sched[sched.scheduler.str.startswith("threads")].iloc[0]
    DODGE = 1.15  # visual dodge only: both series share the same x (the scheduler subset)
    thr_x += [t48.raw_bytes / GiB / DODGE, ceil_thr.raw_bytes / GiB]
    thr_y += [t48.peak_rss_bytes / GiB, ceil_thr.peak_rss_bytes / GiB]
    axA.scatter(thr_x, thr_y, s=30, c=C_THR, marker=M_THR, **ring)

    p48 = sched[sched.scheduler == "processes"]
    px = list(p48.raw_bytes / GiB * DODGE) + [full_raw]
    py = list(p48.peak_rss_bytes / GiB) + [ceil_proc.peak_rss_bytes / GiB]
    axA.scatter(px, py, s=38, c=C_PROC, marker=M_PROC, **ring)

    kw = dict(fontsize=7.8, color=INK)
    lead = dict(arrowstyle="-", color=AXIS, lw=0.7)
    axA.annotate(f"in memory: {slope:.2f} × input\n(R² = {r2:.4f})",
                 xy=(0.6, (slope * 0.6 * GiB + icpt) / GiB), xytext=(0.013, 14),
                 ha="left", va="center", arrowprops=lead, **kw)
    axA.annotate(f"in-memory ceiling:\n{ceiling:.1f} GiB of input", xy=(ceiling, RAM_GIB),
                 xytext=(ceiling / 1.6, 190), ha="right", va="center", arrowprops=lead, **kw)
    right = full_raw * 1.25
    axA.text(right, full_proj, f"full domain\nin memory:\n{full_proj:.0f} GiB (projected)",
             ha="left", va="center", **kw)
    if a.labels == "memory":
        # Peaks to one decimal place, as the paper text quotes them (22.5, 3.8)
        proc_txt = f"full domain,\n{n_proc} processes:\n{ceil_proc.peak_rss_bytes / GiB:.1f} GiB peak"
        thr_txt = f"full domain,\nthreads:\n{ceil_thr.peak_rss_bytes / GiB:.1f} GiB peak"
    else:
        proc_txt = (f"full domain,\n{n_proc} processes:\n{minutes(proc_detect)} min detection,\n"
                    f"{minutes(proc_e2e)} min end to end")
        thr_txt = (f"full domain,\nthreads:\n{minutes(thr_detect)} min detection,\n"
                   f"{minutes(thr_e2e)} min end to end")
    axA.text(right, ceil_proc.peak_rss_bytes / GiB, proc_txt, ha="left", va="center", **kw)
    axA.text(right, ceil_thr.peak_rss_bytes / GiB, thr_txt, ha="left", va="center", **kw)
    if len(sweep):
        sx = sweep.raw_bytes.iloc[0] / GiB
        lo, hi = sweep.mib_per_chunk.min(), sweep.mib_per_chunk.max()
        hi_txt = f"{hi / 1024:.1f} GiB" if hi >= 1024 else f"{hi:.0f} MiB"
        axA.annotate(f"chunk-size sweep:\n{lo:.0f} MiB to {hi_txt} chunks",
                     xy=(sx * 1.1, sweep.peak_rss_bytes.min() / GiB),
                     xytext=(sx * 1.8, 0.9), ha="left", va="center", arrowprops=lead, **kw)

    axA.set_xlim(0.01, 330); axA.set_ylim(0.04, 420)
    ticks = [0.01, 0.1, 1, 10, 100]
    axA.xaxis.set_major_locator(FixedLocator(ticks))
    axA.yaxis.set_major_locator(FixedLocator(ticks))
    for ax_ in (axA.xaxis, axA.yaxis):
        ax_.set_major_formatter(FuncFormatter(gib_fmt)); ax_.set_minor_locator(NullLocator())
    axA.set_xlabel("decoded input size (GiB)", color=INK)
    axA.set_ylabel("peak memory (GiB)", color=INK)
    axA.set_title("A   Memory: in-memory ceiling and out-of-core runs", loc="left",
                  color=INK, fontweight="bold", pad=8)
    handles = [
        Line2D([], [], color=C_MEM, marker=M_MEM, lw=1.4, ms=5.5, mec="white", label="in memory, 1 core"),
        Line2D([], [], color=C_THR, marker=M_THR, lw=0, ms=5.5, mec="white", label="Dask, threads"),
        Line2D([], [], color=C_PROC, marker=M_PROC, lw=0, ms=6.5, mec="white", label="Dask, worker processes"),
        Line2D([], [], color=C_MEM, lw=1.1, ls=(0, (4, 3)), label="in memory, projected"),
    ]
    axA.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.8,
               handlelength=2.2, labelcolor=INK)

    # ------------------------------------------------------------------ B
    style_axes(axB)
    axB.grid(False)
    axB.xaxis.grid(True, color=GRID, linewidth=0.6)
    pr = sched[sched.scheduler == "processes"].set_index("workers")
    rows, _ = reference_rows(ref, a.ref_side, a.xmhw_sync_only)
    t48_rate = col(t48, "detect_cell_steps_per_s")
    if np.isnan(t48_rate):
        t48_rate = t48.cells * t48.timesteps / col(t48, "detect_cold_s", "detect_s")
    rows += [("Subset 33 × 48 × 48, 36 chunks", None, None, None, None),
             ("xrMHW, Dask, 14 threads", t48_rate, C_THR, M_THR, None)]
    for w in (4, 8, 14):
        if w in pr.index:
            rw = pr.loc[w]
            rows.append((f"xrMHW, Dask, {w} processes",
                         rw.cells * rw.timesteps / col(rw, "detect_cold_s", "detect_s"),
                         C_PROC, M_PROC, None))
    full_hdr = (f"Full domain {dims_text(ceil_proc.label)}, "
                f"{gib_label(ceil_proc.raw_bytes)} GiB")
    rows += [
        (full_hdr, None, None, None, None),
        ("xrMHW, Dask, threads", ceil_thr.cells * ceil_thr.timesteps / thr_detect, C_THR, M_THR, None),
        (f"xrMHW, Dask, {n_proc} processes", ceil_proc.cells * ceil_proc.timesteps / proc_detect,
         C_PROC, M_PROC, None),
    ]
    y, ylabels, ypos = 0, [], []
    for label, val, colr, mk, note in rows:
        if val is None:
            y += 0.35 if y else 0
            axB.text(-0.64, -y, label, transform=axB.get_yaxis_transform(), ha="left",
                     va="center", fontsize=8, fontweight="bold", color=INK)
            y += 0.9
            continue
        v = val / 1e6
        axB.scatter([v], [-y], s=40 if mk != M_PROC else 48, c=colr, marker=mk, zorder=4,
                    edgecolors="white", linewidths=0.9)
        txt = sig3(v) + (f"  ({note})" if note else "")
        axB.text(v * 1.28, -y, txt, va="center", ha="left", fontsize=7.8, color=INK)
        ylabels.append(label); ypos.append(-y)
        y += 1
    axB.set_yticks(ypos); axB.set_yticklabels(ylabels, color=INK)
    axB.tick_params(axis="y", length=0, pad=6)
    axB.set_ylim(-y + 0.4, 0.8)
    axB.set_xscale("log"); axB.set_xlim(0.001 if min(r[1] for r in rows if r[1]) < 1e4 else 0.01, 100)
    lo = axB.get_xlim()[0]
    axB.xaxis.set_major_locator(FixedLocator([t for t in (0.001, 0.01, 0.1, 1, 10, 100) if t >= lo]))
    axB.xaxis.set_major_formatter(FuncFormatter(gib_fmt)); axB.xaxis.set_minor_locator(NullLocator())
    axB.set_xlabel("detection throughput (million grid-cell time steps per second)", color=INK)
    axB.set_title("B   Detection throughput", loc="left", x=-0.64, color=INK,
                  fontweight="bold", pad=8)

    for ext, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(f"{out}.{ext}", dpi=dpi, facecolor="white")
    print(f"slope {slope:.4f}  R2 {r2:.5f}  ceiling {ceiling:.2f} GiB  "
          f"full-domain projected {full_proj:.1f} GiB  -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
