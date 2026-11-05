# xrMHW benchmarks

The scripts and results behind the Performance section and Figure 1 of the xrMHW
paper.

| File | What it is |
|---|---|
| `benchmark_xrMHW_v3_2.py` | The benchmark behind the paper's 22-Sep-2026 draft. Five phases, each writing its own CSV (see below). Kept unchanged so those numbers stay reproducible. |
| `make_fig_performance.py` | Draws the 22-Sep-2026 Figure 1 from the `bench_v2*` CSV files. |
| `bench_v2/` | 22-Sep-2026 results with Dask's threaded scheduler, and the in-memory and reference phases. |
| `bench_v2_proc/` | 22-Sep-2026 results with a `LocalCluster` of worker processes (one thread each). |
| `fig_performance.png` | Figure 1 of the 22-Sep-2026 draft (`make_fig_performance.py` also writes a PDF). |
| `benchmark_xrMHW_v3_3.py` | Revised benchmark: three-pass timing, a fair xmhw comparison, a separate scratch folder (see below). Same phases and options as v3_2. |
| `bench_v3/` | Results of `benchmark_xrMHW_v3_3.py` with threads (ceiling) and of the reference phase. |
| `bench_v3_proc/` | Results of `benchmark_xrMHW_v3_3.py` with worker processes (ceiling). |

## Setup used for the paper

- Data: GLORYS12V1 daily potential temperature (`thetao`), 1993–2025, over the
  Grand Banks of Newfoundland (42–50°N, 54–45°W; 97 × 109 cells at 1/12°; 33 levels
  to 644 m), 31 GiB decoded. Climatology 1993–2022. E.U. Copernicus Marine Service
  Information, https://doi.org/10.48670/moi-00021.
- Machine: laptop with an Intel Core i7-13700H (14 cores), 64 GB of RAM, reading from
  an external USB drive.

The input file is not distributed here; download the same product and region from
the Copernicus Marine Data Store to reproduce the runs.

## Phases

| Phase | Measures |
|---|---|
| `scaling` | In-memory runs of increasing size: single-core throughput and peak memory relative to the decoded input |
| `sweep` | Chunk sizes on one 4-D subset, detection and write timed separately |
| `scheduler` | Dask threads against worker processes on the same subset |
| `ceiling` | The full domain, too large for memory, processed out of core |
| `reference` | xrMHW, marineHeatWaves (Oliver) and xmhw on the same 256 surface cells |

## Running

Run the scripts from an activated conda environment (`conda activate <env>`). Calling
the environment's `python.exe` directly without activating it can crash NumPy's
linear algebra with MKL on Windows (`0xC06D007F`), which `np.polyfit` in the scaling
phase and in the figure scripts uses.

The calls used for the 22-Sep-2026 draft:

```bash
python benchmark_xrMHW_v3_2.py --input GLORYS_GrandBanks_1993_2025.nc --var thetao \
    --clim 1993 2022 --phase all --outdir bench_v2

# worker processes and Zarr output for the full domain
python benchmark_xrMHW_v3_2.py --input GLORYS_GrandBanks_1993_2025.nc --var thetao \
    --phase ceiling --ceiling-workers 14 --write-format zarr --outdir bench_v2_proc

python make_fig_performance.py --bench bench_v2 --bench-proc bench_v2_proc
```

The revised runs (`--scratch` takes the temporary output, which is written and deleted
in each measurement; put it on a disk with room for it):

```bash
python benchmark_xrMHW_v3_3.py --input GLORYS_GrandBanks_1993_2025.nc --var thetao \
    --clim 1993 2022 --phase reference --ref-side 16 --xmhw-scheduler both --outdir bench_v3

python benchmark_xrMHW_v3_3.py --input GLORYS_GrandBanks_1993_2025.nc --var thetao \
    --clim 1993 2022 --phase ceiling --chunk-side 4 --write-format zarr \
    --outdir bench_v3 --scratch <scratch>

python benchmark_xrMHW_v3_3.py --input GLORYS_GrandBanks_1993_2025.nc --var thetao \
    --clim 1993 2022 --phase ceiling --ceiling-workers 14 --chunk-side 8 \
    --write-format zarr --outdir bench_v3_proc --scratch <scratch>
```

Run `python <script> --help` for every option. For the reference phase, `xmhw` must
be installed; `marineHeatWaves.py` is taken from the environment or, failing that,
from the vendored copy in `tests/`, and the file used is recorded in the JSON.

## Timing: three passes (v3_3)

`benchmark_xrMHW_v3_2.py` timed detection once (a reduction, reading the input
cold) and then detection plus write (reading it warm, from the operating system's
file cache), and took the write as the difference. That mixed a cold and a warm
read: with worker processes the second pass finished before the first, so the
write time came out negative.

`benchmark_xrMHW_v3_3.py` runs three passes over the same chunked input, reusing
the cluster:

| Pass | Cache | What it does | Column |
|---|---|---|---|
| 0 | cold | detection only | `detect_cold_s` |
| 1 | warm | detection only | `detect_warm_s` |
| 2 | warm | detection and write | `total_warm_s` |

`write_s = total_warm_s − detect_warm_s` compares two warm passes;
`write_resolved` flags whether it exceeds 2% of `total_warm_s`;
`end_to_end_est_s = detect_cold_s + max(write_s, 0)` estimates one run that reads
from disk and writes. `detect_s` and `total_s` keep their v3_2 meaning (passes 0 and 2).
"Cold" means the first read of the invocation; if the file was read shortly before,
it is already cached and `cold_over_warm` is close to 1. Disconnecting and
reconnecting the drive, or rebooting, empties the cache.

Both schedulers write Zarr in the revised ceiling runs, so the two full-domain
results are measured the same way.

## Reading the results

- Timings depend on the machine, the disk and the chunking; compare configurations
  run on the same machine.
- In `bench_v2*`, the write time is the difference between a warm and a cold pass and
  underestimates the write; rows where it is under 2% of the total are flagged
  `write_resolved = False`. `bench_v3*` use the three-pass decomposition above.
- With worker processes, writing one large netCDF file failed after about 5 GB
  (`NetCDF: Unknown file format`); the full-domain runs therefore write Zarr.
- In the 22-Sep-2026 reference phase, xmhw ran on Dask's threaded scheduler with 14
  threads while xrMHW and marineHeatWaves ran on one core. `bench_v3` times xmhw both
  on one thread (`synchronous`) and on the threaded scheduler.
