# bench_v3 results for the paper

The values that fill the `{{…}}` placeholders of the paper draft (`paper_v4_draft.md`) and
of the README, with their source and formula. Runs of 5 November 2026 on host `Gabo`
(Intel Core i7-13700H, 14 physical cores, 63.7 GiB RAM, Windows), xrMHW 3.0.0
(editable install), input read from an external USB drive (E:), temporary output on E:.

| Run | Script and phase | Results |
|---|---|---|
| 1 | `benchmark_xrMHW_v3_3.py --phase reference --ref-side 16 --xmhw-scheduler both` | `bench_v3_reference_Gabo.csv/.json` |
| 2 | `benchmark_xrMHW_v3_3.py --phase ceiling --chunk-side 4 --write-format zarr` (threads) | `bench_v3_ceiling_Gabo.csv/.json` |
| 3 | `benchmark_xrMHW_v3_3.py --phase ceiling --ceiling-workers 14 --chunk-side 8 --write-format zarr` | `../bench_v3_proc/bench_v3_ceiling_Gabo.csv/.json` |
| 4 | `bench_storage.py --workers 14 --chunk-side 8` | `bench_v3_storage_Gabo.csv/.json` |

Full domain: 33 × 97 × 109 = 348,909 cells × 12,017 days = 4.193 × 10⁹ grid-cell time
steps, 31.24 GiB decoded (float64). Reference subset: 16 × 16 = 256 surface cells.
GiB = 2³⁰ bytes; minutes = seconds / 60.

## Placeholder values

| Placeholder | In the text | Exact | Source (file: column) | Formula |
|---|---|---|---|---|
| `{{XMHW_RATIO_SYNC}}` | 93.6 | 93.64 | reference: `seconds` | xmhw [grid, synchronous] 120.07 s / xrMHW 1.282 s |
| `{{XMHW_RATIO_THREADS}}` | 113 | 112.91 | reference: `seconds` | xmhw [grid, threads] 144.77 s / xrMHW 1.282 s |
| `{{XMHW_THRESH_FRAC}}` | 91 | 90.64 | reference: `threshold_s`, `seconds` | 100 × 108.84 s / 120.07 s (synchronous run; 90.2 with threads) |
| `{{THR_DETECT_COLD}}` | 11.5 | 11.487 | ceiling (threads): `detect_cold_s` | 689.23 s / 60 |
| `{{THR_WRITE}}` | 1.4 | 1.370 | ceiling (threads): `write_s` | 82.19 s / 60 (resolved: 11% of `total_warm_s`) |
| `{{THR_E2E}}` | 12.9 | 12.857 | ceiling (threads): `end_to_end_est_s` | (689.23 + 82.19) s / 60 |
| `{{THR_PEAK}}` | 3.8 | 3.784 | ceiling (threads): `peak_rss_bytes` | 4,062,568,448 B / 2³⁰ |
| `{{THR_RATE}}` | 6.08 | 6.083 | ceiling (threads): `detect_cold_s` | 4.193 × 10⁹ / 689.23 s / 10⁶ (panel B) |
| `{{PROC_DETECT_COLD}}` | 4.1 | 4.118 | ceiling (processes): `detect_cold_s` | 247.09 s / 60 |
| `{{PROC_WRITE}}` | 0.05 | 0.047 | ceiling (processes): `write_s` | 2.82 s / 60; **not resolved**, see note 4 |
| `{{PROC_E2E}}` | 4.2 | 4.165 | ceiling (processes): `end_to_end_est_s` | (247.09 + 2.82) s / 60 |
| `{{PROC_PEAK}}` | 22.5 | 22.483 | ceiling (processes): `peak_rss_bytes` | 24,140,754,944 B / 2³⁰ (figure label: 22.5) |
| `{{PROC_RATE}}` | 17.0 | 16.969 | ceiling (processes): `detect_cold_s` | 4.193 × 10⁹ / 247.09 s / 10⁶ |
| `{{PROC_SPEEDUP}}` | 6.95 | 6.954 | — | 16.969 / 2.44 |
| `{{NC_RAW}}` | 17.2 | 17.208 | storage: case `a`, `bytes` | 18,476,708,123 B / 2³⁰; see note 5 |
| `{{NC_ZLIB}}` | 1.29 | 1.285 | storage: case `b`, `bytes` | 1,379,845,315 B / 2³⁰ |
| `{{ZARR}}` | 1.13 | 1.129 | storage: case `c`, `bytes` | 1,212,077,932 B / 2³⁰ |
| `{{ZARR_CODEC}}` | Zstandard | — | storage: `codec` | `ZstdCodec(level=0, checksum=False)`, zarr-python 3.1.5 default, Zarr format 3; level 0 is Zstd's default level, 3 |
| `{{STORAGE_RATIO}}` | 15 | 15.24 | storage: `ratio_nc_raw_over_zarr` | 17.208 / 1.129 (13.39 against netCDF-4 with zlib: 17.208 / 1.285) |
| `{{EVENT_FRACTION}}` | 9.4% | 9.353% | storage: `event_fraction_of_ocean_cell_days` | 325,817,171 event cell-days / (289,878 ocean cells × 12,017 days) |
| `{{V3_DOI}}` | — | — | — | not available until the 3.0.0 release is archived on Zenodo; left as a placeholder |

`{{THR_E2E}}`, `{{PROC_E2E}}`, `{{THR_RATE}}`, `{{ZARR_CODEC}}` and `{{EVENT_FRACTION}}`
do not appear in the draft text; they are used in the figure or kept here for reference.

## Checks on the runs

- **Cold cache.** `cold_over_warm` is 1.036 with threads and 1.448 with processes. The
  drive was ejected and reconnected before run 3, and the processes run shows the cold read
  clearly. With threads the computation dominates, so a value close to 1 is expected.
- **Writes.** No write errors. The threads write (82.2 s) is resolved. The processes write
  (2.82 s) is under the 2% resolution limit (3.47 s of `total_warm_s` = 173.4 s), so
  `write_resolved = False`.
- **Event counts, 256 cells.** xrMHW and marineHeatWaves agree in all 256 cells (19,984
  events each). xrMHW and xmhw differ in 22 of 256 cells, by at most 2 events, with
  19,984 against 19,985 events in total.
- **Storage.** Zarr chose `ZstdCodec(level=0)`. All four cases use the same chunks,
  `{depth: 33, latitude: 8, longitude: 8, time: 12017}`. Copies a–c read a source store
  that had just been written, so their times are write times from a warm source: netCDF
  uncompressed 225 s, netCDF zlib 53 s, Zarr 9 s.

## Changes from the 22-Sep-2026 draft (bench_v2)

| Quantity | bench_v2 | bench_v3 | Change |
|---|---|---|---|
| xrMHW, 256 cells | 1.260 s | 1.282 s | +2% |
| marineHeatWaves, 256 cells | 23.71 s (18.8×) | 20.02 s (15.6×) | −16% |
| xmhw, 256 cells | 142.85 s, 14 threads (113.4×) | 144.77 s, threads (112.9×); 120.07 s, synchronous (93.6×) | threads +1% |
| Threads, full-domain detection | 901.0 s (15.0 min) | 689.2 s (11.5 min) | −24% |
| Threads, peak | 3.45 GiB | 3.78 GiB | +9.6% |
| Threads, write | netCDF, 672.9 s (total − detect, cold/warm mixed) | Zarr, 82.2 s | not comparable |
| Processes, full-domain detection | 269.1 s (4.5 min) | 247.1 s (4.1 min) | −8% |
| Processes, peak | 21.76 GiB | 22.48 GiB | +3.3% |
| Processes / threads, detection | 3.35× | 2.79× | — |
| Zarr output, severity | 1.13 GiB | 1.13 GiB (1,212,077,932 B, identical) | 0 |

## Notes for the text

1. **marineHeatWaves ratio.** The agreed text (Appendix A.7 of the brief) said "18.8
   times", from bench_v2. The bench_v3 reference run gives **15.6 times** (20.02 s /
   1.282 s), as panel B of the new figure shows, and the draft now says 15.6, so all three
   ratios (15.6, 93.6, 113) come from one run.
2. **Threaded detection is faster than in bench_v2** (11.5 against 15.0 min); detection
   with processes changed less (−8%). The cause was not investigated; a different power
   mode or background load during the September run are possible. The draft states no
   threads-to-processes ratio, so no sentence contradicts it.
3. **Peaks.** Threads +9.6% (3.45 → 3.78 GiB), just under the 10% flag; processes +3.3%.
4. **`{{PROC_WRITE}}` = 0.05 min is within measurement noise.** The difference between
   the two warm passes, 2.8 s, is below the 2% resolution limit. A wording that does not
   imply precision would be "and writing added only a few seconds".
5. **Uncompressed netCDF is padded.** 17.2 GiB is 10% more than the 15.62 GiB of float32
   data. Chunks of 8 do not divide 97 or 109, so the 13 × 14 chunks cover 104 × 112
   cells, and HDF5 allocates the edge chunks in full: 33 × 104 × 112 × 12,017 × 4 B =
   18,476,570,112 B, which matches the file to within its metadata. Measured against the
   data size, Zarr is 13.8 times smaller rather than 15.2. Both numbers are measured;
   choose which one the text should quote.
6. **xmhw event counts** differ from xrMHW in 22 of 256 cells (at most 2 events per
   cell). The draft makes no claim about agreement with xmhw; the agreement it does claim,
   with marineHeatWaves, holds in all 256 cells.
7. **Single-core rate.** In panel B, xrMHW on the 256 reference cells runs at 2.40 million
   grid-cell time steps per second. The text's 2.44 is the mean of the scaling ladder
   (bench_v2). The two agree within the ladder's 10% coefficient of variation.
