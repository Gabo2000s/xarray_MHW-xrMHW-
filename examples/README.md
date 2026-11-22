# xrMHW example notebooks

Six short notebooks that show how to detect marine heatwaves with xrMHW, from a single
time series to a four-dimensional ocean reanalysis. Read them in order the first time.

| Notebook | What it covers |
|---|---|
| [00 · Setup and example data](00_setup_and_data.ipynb) | Installation check, downloading the example data, what xrMHW needs from your data |
| [01 · A single time series](01_single_point_1D.ipynb) | Detection at one point, the meaning of each output, an event table, validation against `marineHeatWaves` |
| [02 · Sea surface maps](02_sea_surface_3D.ipynb) | Detection on a latitude–longitude grid, maps of one day, event days per year, marine cold spells |
| [03 · Subsurface marine heatwaves](03_subsurface_4D.ipynb) | Detection over depth, latitude and longitude; profiles, depth–time diagrams, sections, events hidden below the surface, a 3-D view |
| [04 · Many files, Dask and Zarr](04_many_files_and_scaling.ipynb) | `open_mfdataset`, choosing chunk sizes from memory, a local Dask cluster, writing to Zarr |
| [05 · Troubleshooting](05_troubleshooting.ipynb) | Common errors and silent pitfalls, reproduced, with their fixes |

## Running them

From the repository root, either with conda (cartopy installs most easily this way):

```bash
conda env create -f examples/environment.yml
conda activate xrmhw-examples
pip install -e .
```

or with pip:

```bash
pip install ".[examples]"
```

Then start Jupyter in this folder (`jupyter lab`) and open `00_setup_and_data.ipynb`.
The notebooks write their outputs to `examples/xrmhw_outputs/`, which git ignores.

## Example data

The notebooks download two files, 86 MB in total, from Zenodo with
[pooch](https://www.fatiando.org/pooch) the first time they run, and reuse the cached
copies afterwards:

- `xrmhw_glorys_4d_subset.nc`: GLORYS12V1 daily potential temperature, 1993–2025,
  33 levels to 644 m, 12 × 12 cells on the south-eastern slope of the Grand Bank of
  Newfoundland.
- `xrmhw_ostia_sst_3d.nc`: OSTIA daily sea surface temperature, 1990–2022, off the
  Pacific coast of Baja California.

Record: https://doi.org/10.5281/zenodo.22909109. The files are E.U. Copernicus Marine
Service Information (https://doi.org/10.48670/moi-00021,
https://doi.org/10.48670/moi-00168), redistributed under the Copernicus Marine
Service licence. [`make_example_data.py`](make_example_data.py) is the script that
produced them from the Copernicus downloads.

## For maintainers

The notebooks are stored with their outputs so that figures show on GitHub. After
changing a notebook or xrMHW, run them again from this folder and commit the result:

```bash
jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

The workflow in `.github/workflows/notebooks.yml` runs every notebook on each change
to `examples/` or `xrMHW.py`, and once a month, and fails if any cell raises an error.
