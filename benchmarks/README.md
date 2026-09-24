# xrMHW benchmarks

The scripts and results behind the Performance section and Figure 1 of the xrMHW
paper.

| File | What it is |
|---|---|
| `benchmark_xrMHW_v3_2.py` | The benchmark. Five phases, each writing its own CSV (see below). |
| `make_fig_performance.py` | Draws Figure 1 from the CSV files. |
| `bench_v2/` | Results with Dask's threaded scheduler, and the in-memory and reference phases. |
| `bench_v2_proc/` | Results with a `LocalCluster` of worker processes (one thread each). |
| `fig_performance.png` | Figure 1 as it appears in the paper (`make_fig_performance.py` also writes a PDF). |

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

Examples of the calls used for the paper:

```bash
python benchmark_xrMHW_v3_2.py --input GLORYS_GrandBanks_1993_2025.nc --var thetao \
    --clim 1993 2022 --phase all --outdir bench_v2

# worker processes and Zarr output for the full domain
python benchmark_xrMHW_v3_2.py --input GLORYS_GrandBanks_1993_2025.nc --var thetao \
    --phase ceiling --ceiling-workers 14 --write-format zarr --outdir bench_v2_proc

python make_fig_performance.py --bench bench_v2 --bench-proc bench_v2_proc
```

Run `python benchmark_xrMHW_v3_2.py --help` for every option. For the reference
phase, `marineHeatWaves.py` must be importable (see `tests/`) and `xmhw` installed.

## Reading the results

- Timings depend on the machine, the disk and the chunking; compare configurations
  run on the same machine.
- The write time is measured as a second pass after detection. By then the operating
  system has cached the input, so subtracting detection from the total underestimates
  the write; rows where the difference is under 2% of the total are flagged
  `write_resolved = False`.
- With worker processes, writing one large netCDF file failed after about 5 GB
  (`NetCDF: Unknown file format`); the full-domain runs therefore write Zarr.
