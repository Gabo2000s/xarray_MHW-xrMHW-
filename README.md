# MHW_xarray_multidimensional_matrix
High-performance Marine Heatwave (MHW) detector for multidimensional ocean data. Implements the standard Hobday et al. (2016) definition using Python, Xarray, and Dask for parallel processing.

# Multidimensional Marine Heatwave Detector (Xarray + Dask)

A robust, high-performance Python implementation for detecting Marine Heatwaves (MHW) in multidimensional oceanographic datasets (3D and 4D).

This tool implements the standard definition proposed by **Hobday et al. (2016)**, leveraging the power of **Xarray** and **Dask** to parallelize detection across vast spatial grids (Latitude, Longitude, Depth) without running out of memory.

## 🌊 Key Features

* **Multidimensional Support:** Native handling of 4D data (Time, Depth, Lat, Lon). Ideal for reanalysis products like **GLORYS**, **ORAS5**, or **OISST**.
* **Parallelized Efficiency:** Uses `xarray.apply_ufunc` with Dask backend to vectorize operations, avoiding slow Python loops over spatial coordinates.
* **Scientific Rigor:**
    * Calculates a 30-year climatological baseline (default).
    * Uses a 90th percentile threshold with an 11-day centered window.
    * Enforces a 5-day minimum duration for event detection.
* **Robustness:** Automatically handles leap years, missing data (NaNs), and land masks.

## 📦 Requirements

* Python 3.8+
* xarray
* dask
* numpy
* pandas
* scipy
* netCDF4

Install dependencies using pip:

```bash
pip install -r requirements.txt
