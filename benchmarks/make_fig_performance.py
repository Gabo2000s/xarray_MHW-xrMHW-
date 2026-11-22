"""Figure 1 of the xrMHW paper: memory (A) and detection throughput (B).

Reads the CSVs written by benchmark_xrMHW_v3_2.py. Run from the scripts folder:
    python make_fig_performance.py --bench bench_v2 --bench-proc bench_v2_proc --out fig_performance
"""
import argparse
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bench_v2")
    ap.add_argument("--bench-proc", default="bench_v2_proc")
    ap.add_argument("--host", default="Gabo")
    ap.add_argument("--out", default="fig_performance")
    a = ap.parse_args()
    b, bp, h = Path(a.bench), Path(a.bench_proc), a.host

    scal = pd.read_csv(b / f"bench_v2_scaling_{h}.csv")
    sweep = pd.read_csv(b / f"bench_v2_sweep_{h}.csv")
    ref = pd.read_csv(b / f"bench_v2_reference_{h}.csv")
    ceil_thr = pd.read_csv(b / f"bench_v2_ceiling_{h}.csv").iloc[0]
    sched = pd.read_csv(bp / f"bench_v2_scheduler_{h}.csv")
    ceil_proc = pd.read_csv(bp / f"bench_v2_ceiling_{h}.csv").iloc[0]

    # in-memory fit on subsets >= 64 MiB, as in the benchmark
    fit = scal[scal.raw_bytes >= 64 * 2 ** 20]
    slope, icpt = np.polyfit(fit.raw_bytes, fit.peak_rss_bytes, 1)
    r2 = np.corrcoef(fit.raw_bytes, fit.peak_rss_bytes)[0, 1] ** 2
    ceiling = (RAM_GIB * GiB - icpt) / slope / GiB
    full_raw = ceil_proc.raw_bytes / GiB
    full_proj = (slope * ceil_proc.raw_bytes + icpt) / GiB

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig = plt.figure(figsize=(6.3, 7.6))
    axA = fig.add_axes([0.10, 0.555, 0.86, 0.405])
    axB = fig.add_axes([0.385, 0.075, 0.575, 0.36])

    # ------------------------------------------------------------------ A
    style_axes(axA)
    axA.set_xscale("log"); axA.set_yscale("log")
    xs = np.geomspace(0.05, scal.raw_bytes.max() / GiB, 50)
    axA.plot(xs, (slope * xs * GiB + icpt) / GiB, color=C_MEM, lw=1.4, zorder=2)
    xp = np.geomspace(scal.raw_bytes.max() / GiB, full_raw, 50)
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
    DODGE = 1.15  # visual dodge only: both series share the same x (the 33x48x48 subset)
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
    proj_txt = f"{full_proj:.0f}"
    right = full_raw * 1.25
    axA.text(right, full_proj, f"full domain\nin memory:\n{proj_txt} GiB (projected)",
             ha="left", va="center", **kw)
    axA.text(right, ceil_proc.peak_rss_bytes / GiB,
             f"full domain,\n14 processes:\n{ceil_proc.detect_s / 60:.1f} min detection",
             ha="left", va="center", **kw)
    axA.text(right, ceil_thr.peak_rss_bytes / GiB,
             f"full domain,\nthreads:\n{ceil_thr.total_s / 60:.0f} min end to end",
             ha="left", va="center", **kw)
    sx = sweep.raw_bytes.iloc[0] / GiB
    axA.annotate("chunk-size sweep:\n27 MiB to 1.7 GiB chunks", xy=(sx * 1.1, 2.25),
                 xytext=(sx * 1.8, 0.9), ha="left", va="center", arrowprops=lead, **kw)

    axA.set_xlim(0.01, 330); axA.set_ylim(0.04, 420)
    ticks = [0.01, 0.1, 1, 10, 100]
    axA.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
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
    r = ref.set_index(ref.tool.str.split(" ").str[0])
    pr = sched[sched.scheduler == "processes"].set_index("workers")
    xr_one = r.loc["xrMHW", "cell_steps_per_s"]
    rows = [
        ("Same 256 surface cells", None, None, None, None),
        ("xmhw", r.loc["xmhw", "cell_steps_per_s"], C_REF, "D", f"{xr_one / r.loc['xmhw', 'cell_steps_per_s']:.0f}× slower"),
        ("marineHeatWaves, cell by cell", r.loc["marineHeatWaves", "cell_steps_per_s"], C_REF, "D",
         f"{xr_one / r.loc['marineHeatWaves', 'cell_steps_per_s']:.1f}× slower"),
        ("xrMHW, in memory, 1 core", xr_one, C_MEM, M_MEM, None),
        ("Subset 33 × 48 × 48, 36 chunks", None, None, None, None),
        ("xrMHW, Dask, 14 threads", t48.detect_cell_steps_per_s if "detect_cell_steps_per_s" in t48 and pd.notna(t48.get("detect_cell_steps_per_s")) else t48.cells * t48.timesteps / t48.detect_s, C_THR, M_THR, None),
    ]
    for w in (4, 8, 14):
        rw = pr.loc[w]
        rows.append((f"xrMHW, Dask, {w} processes", rw.cells * rw.timesteps / rw.detect_s, C_PROC, M_PROC, None))
    rows += [
        ("Full domain 33 × 97 × 109, 31 GiB", None, None, None, None),
        ("xrMHW, Dask, threads", ceil_thr.cells * ceil_thr.timesteps / ceil_thr.detect_s, C_THR, M_THR, None),
        ("xrMHW, Dask, 14 processes", ceil_proc.cells * ceil_proc.timesteps / ceil_proc.detect_s, C_PROC, M_PROC, None),
    ]
    y, ylabels, ypos = 0, [], []
    for label, val, col, mk, note in rows:
        if val is None:
            y += 0.35 if y else 0
            axB.text(-0.64, -y, label, transform=axB.get_yaxis_transform(), ha="left",
                     va="center", fontsize=8, fontweight="bold", color=INK)
            y += 0.9
            continue
        v = val / 1e6
        axB.scatter([v], [-y], s=40 if mk != M_PROC else 48, c=col, marker=mk, zorder=4,
                    edgecolors="white", linewidths=0.9)
        txt = (f"{v:.3f}" if v < 1 else f"{v:.3g}") + (f"  ({note})" if note else "")
        axB.text(v * 1.28, -y, txt, va="center", ha="left", fontsize=7.8, color=INK)
        ylabels.append(label); ypos.append(-y)
        y += 1
    axB.set_yticks(ypos); axB.set_yticklabels(ylabels, color=INK)
    axB.tick_params(axis="y", length=0, pad=6)
    axB.set_ylim(-y + 0.4, 0.8)
    axB.set_xscale("log"); axB.set_xlim(0.01, 100)
    axB.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
    axB.xaxis.set_major_formatter(FuncFormatter(gib_fmt)); axB.xaxis.set_minor_locator(NullLocator())
    axB.set_xlabel("detection throughput (million grid-cell time steps per second)", color=INK)
    axB.set_title("B   Detection throughput", loc="left", x=-0.64, color=INK,
                  fontweight="bold", pad=8)

    for ext, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(f"{a.out}.{ext}", dpi=dpi, facecolor="white")
    print(f"slope {slope / 1:.4f}  R2 {r2:.5f}  ceiling {ceiling:.2f} GiB  full-domain projected {full_proj:.1f} GiB")


if __name__ == "__main__":
    main()
