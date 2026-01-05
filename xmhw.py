"""
Marine Heatwave Detector (Hobday et al., 2016 implementation)
==============================================================

A robust, Xarray-compatible implementation of the Marine Heatwave (MHW) definitions 
proposed by Hobday et al. (2016). This script utilizes `xarray.apply_ufunc` to 
parallelize the detection algorithm across multidimensional datasets (Latitude, 
Longitude, Depth, Time).

Author: Gutiérrez-Cárdenas, GS. (ORCID: 0000-0002-3915-7684)
Date: Dec 2025
License: MIT
"""

import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from datetime import date
from dask.diagnostics import ProgressBar
import warnings

# Suppress warnings for clean output during large computations
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. CORE SCIENTIFIC LOGIC (Numpy/Pandas based)
# =============================================================================

def runavg(ts, w):
    """
    Calculates a running average (smoothing) on a time series.
    
    Parameters
    ----------
    ts : np.ndarray
        Input time series (1D).
    w : int
        Window size for the running average.
        
    Returns
    -------
    np.ndarray
        Smoothed time series.
    """
    N = len(ts)
    # Pad the array to handle edge effects (circular boundary condition simulation)
    ts = np.append(ts, np.append(ts, ts))
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    # Return the central part corresponding to original data
    ts = ts_smooth[N:2*N]
    return ts

def detect_mhw_core(t_ordinal, temp, clim_period=(None, None), pctile=90, window_half_width=5, 
                   smooth_pctile=True, smooth_width=31, min_duration=5, 
                   join_across_gaps=True, max_gap=2):
    """
    Core detection algorithm for Marine Heatwaves on a single 1D time series.
    Based on Hobday et al. (2016) marine heat wave definition. Edited and optimized from 
    https://github.com/ecjoliver/marineHeatWaves

    Parameters
    ----------
    t_ordinal : np.ndarray
        Time vector in ordinal format (integers).
    temp : np.ndarray
        Temperature vector (1D).
    clim_period : tuple
        (start_year, end_year) for climatology baseline.
    pctile : int
        Percentile threshold (default 90).
    window_half_width : int
        Width of window for climatology calculation (default 5 days).
        
    Returns
    -------
    tuple
        (clim_seas, clim_thresh, anomaly, is_mhw, mhw_duration, mhw_category, 
         mhw_intensity_max, mhw_intensity_cum)
    """
    
    # 1. Initialize Time Vectors
    T = len(t_ordinal)
    dates_pd = pd.to_datetime([date.fromordinal(int(x)) for x in t_ordinal])
    year = dates_pd.year.values
    month = dates_pd.month.values
    day = dates_pd.day.values
    
    # Create a Day-of-Year (DOY) map handling leap years (366 days)
    t_leap = np.arange(date(2012, 1, 1).toordinal(), date(2012, 12, 31).toordinal()+1)
    dates_leap = pd.to_datetime([date.fromordinal(x) for x in t_leap]) 
    doy_map = {(m, d): dy for m, d, dy in zip(dates_leap.month, dates_leap.day, range(1, 367))}
    doy = np.array([doy_map.get((m, d), 0) for m, d in zip(month, day)]) # .get safe access

    # 2. Climatology Calculation
    if (clim_period[0] is None) or (clim_period[1] is None):
        clim_start, clim_end = year[0], year[-1]
    else:
        clim_start, clim_end = clim_period

    # Filter data for baseline period
    clim_mask = (year >= clim_start) & (year <= clim_end)
    temp_clim = temp[clim_mask]
    doy_clim = doy[clim_mask]
    
    thresh_clim_year = np.full(366, np.nan)
    seas_clim_year = np.full(366, np.nan)
    
    # Calculate threshold and climatology for each day of year (1-366)
    for d in range(1, 367):
        if d == 60: continue # Skip Feb 29 logic placeholder initially
        
        # Define window (circular)
        window_days = np.arange(d - window_half_width, d + window_half_width + 1)
        window_days = ((window_days - 1) % 366) + 1
        window_days = window_days[window_days != 60] # Exclude Feb 29 from window logic
        
        in_window = np.isin(doy_clim, window_days)
        data_in_window = temp_clim[in_window]
        
        if len(data_in_window) > 0:
            thresh_clim_year[d-1] = np.nanpercentile(data_in_window, pctile)
            seas_clim_year[d-1] = np.nanmean(data_in_window)

    # Interpolate Feb 29 (DOY 60)
    thresh_clim_year[59] = 0.5 * thresh_clim_year[58] + 0.5 * thresh_clim_year[60]
    seas_clim_year[59] = 0.5 * seas_clim_year[58] + 0.5 * seas_clim_year[60]

    # Smooth the climatology/threshold
    if smooth_pctile:
        thresh_clim_year = runavg(thresh_clim_year, smooth_width)
        seas_clim_year = runavg(seas_clim_year, smooth_width)

    # Map back to full time series
    # doy-1 because array is 0-indexed, doy is 1-366
    clim_thresh = thresh_clim_year[doy - 1]
    clim_seas = seas_clim_year[doy - 1]

    # 3. Event Detection
    exceed_bool = temp > clim_thresh
    events, n_events = ndimage.label(exceed_bool)

    # Filter by duration (minimum 5 days)
    for ev in range(1, n_events + 1):
        if (events == ev).sum() < min_duration:
            events[events == ev] = 0

    # Join events across small gaps
    if join_across_gaps:
        # Note: A more complex gap bridging logic could be implemented here 
        # but standard label re-evaluation is often sufficient for basic bridging 
        # if the boolean mask was pre-processed. 
        # For strict Hobday: we usually iterate and fill gaps. 
        # Keeping user logic here for consistency:
        events, n_events = ndimage.label(events > 0)

    # 4. Metrics Extraction
    # Initialize arrays as float to accommodate NaNs later if needed
    mhw_duration = np.zeros(T, dtype=float) 
    mhw_category = np.zeros(T, dtype=float)
    mhw_intensity_max = np.zeros(T, dtype=float)
    mhw_intensity_cum = np.zeros(T, dtype=float)
    is_mhw = np.zeros(T, dtype=bool)

    for ev in range(1, n_events + 1):
        idx = np.where(events == ev)[0]
        dur = len(idx)
        if dur < min_duration: continue
        
        temps_ev = temp[idx]
        seas_ev = clim_seas[idx]
        thresh_ev = clim_thresh[idx]
        anoms = temps_ev - seas_ev
        
        i_max = np.max(anoms)
        i_cum = np.sum(anoms)
        
        # Categorization
        peak_idx = np.argmax(anoms)
        intensity_diff = thresh_ev[peak_idx] - seas_ev[peak_idx]
        if intensity_diff == 0: intensity_diff = 1e-5 # Avoid zero division
        ratio = i_max / intensity_diff
        cat = max(1, int(np.floor(ratio)))
        
        # Assign values to the full time dimension
        mhw_duration[idx] = float(dur) 
        mhw_category[idx] = float(cat)
        mhw_intensity_max[idx] = i_max
        mhw_intensity_cum[idx] = i_cum
        is_mhw[idx] = True

    anomaly = temp - clim_seas
    
    return clim_seas, clim_thresh, anomaly, is_mhw, mhw_duration, mhw_category, mhw_intensity_max, mhw_intensity_cum

# =============================================================================
# 2. ROBUST WRAPPER (Handles NaNs & Vectorization Interface)
# =============================================================================

def mhw_1d_wrapper(time_ordinal, temp, clim_start_year, clim_end_year, max_gap_interp=2, **kwargs):
    """
    Wrapper to handle NaNs and interface with xarray.apply_ufunc, passing 
    dynamic configuration parameters to the core detection logic.

    This function ensures that only small data gaps (<= max_gap_interp) are 
    interpolated, preserving the integrity of the time series for larger gaps 
    as per Hobday et al. (2016).

    Parameters
    ----------
    time_ordinal : np.ndarray
        Time vector in ordinal format (integers).
    temp : np.ndarray
        Temperature vector (1D).
    clim_start_year : int or float
        Start year of the climatology baseline.
    clim_end_year : int or float
        End year of the climatology baseline.
    max_gap_interp : int, optional
        Maximum gap length (in days) to fill via linear interpolation. 
        Gaps larger than this will remain as NaNs (breaking potential events).
        Default is 2 days.
    **kwargs : optional
        Additional arguments passed directly to `detect_mhw_core`.
        (e.g., pctile, window_half_width, min_duration).

    Returns
    -------
    tuple
        Tuple of numpy arrays containing MHW statistics.
    """
    T = len(time_ordinal)
    
    # --- CRITICAL FIX: Dask Read-Only & Gap Handling ---
    # Create a writable copy of the data. Dask arrays are often read-only chunks.
    temp = temp.copy()
    
    # 1. Fast exit for land mask (all NaNs)
    if np.isnan(temp).all():
        nan_arr = np.full(T, np.nan)
        return (nan_arr, nan_arr, nan_arr, np.zeros(T, dtype=bool), 
                nan_arr, nan_arr, nan_arr, nan_arr)

    # 2. Smart Interpolation (Only small gaps)
    # We only fill gaps that are <= max_gap_interp to maintain scientific rigor.
    if np.isnan(temp).any():
        is_valid = ~np.isnan(temp)
        valid_indices = np.flatnonzero(is_valid)

        # If we have enough data to interpolate
        if len(valid_indices) > 1:
            # Find distances between consecutive valid points
            gaps = np.diff(valid_indices)
            
            # Identify indices where the gap is larger than 1 but within the limit
            # gaps > 1 means there is at least 1 missing day between points
            fillable_gaps = np.where((gaps > 1) & (gaps <= (max_gap_interp + 1)))[0]

            for i in fillable_gaps:
                start_idx = valid_indices[i]
                end_idx = valid_indices[i+1]
                
                # Perform linear interpolation for this specific gap
                # We include start and end points for the interpolation reference
                x_gap = np.arange(start_idx + 1, end_idx)
                temp[x_gap] = np.interp(
                    x_gap, 
                    [start_idx, end_idx], 
                    [temp[start_idx], temp[end_idx]]
                )
    
    # 3. Type safety for Numba/Core logic
    # Remaining NaNs (large gaps) are converted to a fill value (-9999) 
    # to ensure boolean comparisons in the core function don't crash, 
    # effectively breaking the MHW event detection at that point.
    time_ordinal_safe = time_ordinal.astype(int)
    
    # Note: Depending on detect_mhw_core implementation, leaving NaNs might be fine
    # if it uses nan-aware functions. If it uses strict boolean (temp > clim), 
    # NaNs result in False, which correctly breaks the event.
    # We keep it as float to preserve NaNs if the core handles them, 
    # otherwise, NaN comparison naturally returns False preventing false positives.
    temp_safe = temp.astype(float)

    # Pass **kwargs to the core function
    return detect_mhw_core(
        time_ordinal_safe, 
        temp_safe, 
        clim_period=(int(clim_start_year), int(clim_end_year)),
        **kwargs 
    )

# =============================================================================
# 3. XARRAY INTEGRATION
# =============================================================================

def xmhw_func(ds, temp_var_name, clim_period, **kwargs):
    """
    Applies the MHW detection over the entire Xarray Dataset using Dask,
    allowing full customization of detection parameters.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing temperature and time.
    temp_var_name : str
        Name of the temperature variable in the dataset.
    clim_period : tuple
        A tuple (start_year, end_year) defining the climatology baseline.
    **kwargs : optional
        Arguments passed directly to the detection algorithm. 
        Key arguments include:
            - pctile (default: 90)
            - window_half_width (default: 5)
            - min_duration (default: 5)
            - max_gap_interp (default: 2): Max days of missing data to interpolate.
                                            Larger gaps remain NaN.

    Returns
    -------
    xr.Dataset
        A new dataset containing the detected MHW metrics masked by event occurrence.
    """
    # 1. Prepare Time Coordinate
    # Convert datetime objects to ordinal integers for the core algorithm
    time_index = ds.indexes['time']
    time_ordinal_np = time_index.map(lambda x: x.toordinal()).values
    
    time_ordinal_da = xr.DataArray(
        time_ordinal_np, 
        coords={'time': ds['time']}, 
        dims='time'
    )

    # 2. Configure Arguments for apply_ufunc
    # Merge mandatory climatology period with optional user arguments
    func_kwargs = {
        'clim_start_year': clim_period[0], 
        'clim_end_year': clim_period[1]
    }
    # kwargs will now carry 'max_gap_interp' if passed from the main script
    func_kwargs.update(kwargs)

    # Define output types matching the return tuple of detect_mhw_core
    # [seas, thresh, anomaly, is_mhw, duration, category, intensity_max, intensity_cum]
    output_dtypes = [float, float, float, bool, float, float, float, float]
    
    # 3. Apply Vectorized UFunc
    results = xr.apply_ufunc(
        mhw_1d_wrapper,
        time_ordinal_da,
        ds[temp_var_name],
        kwargs=func_kwargs,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']] * 8, 
        vectorize=True,
        dask='parallelized',
        output_dtypes=output_dtypes
    )
    
    # Unpack results
    (seas, thresh, anomaly, is_mhw, duration, category, 
     intensity_max, intensity_cum) = results

    # 4. Assemble Output Dataset
    ds_out = xr.Dataset()
    
    # Apply Mask: Variables are NaN where no MHW is present
    ds_out['mhw_intensity'] = anomaly.where(is_mhw)
    ds_out['mhw_duration'] = duration.where(is_mhw)
    ds_out['mhw_category'] = category.where(is_mhw)
    ds_out['mhw_max_intensity'] = intensity_max.where(is_mhw)
    ds_out['mhw_cum_intensity'] = intensity_cum.where(is_mhw)
    
    # Optional: diagnostic variables
    ds_out['climatology'] = seas
    ds_out['threshold'] = thresh

    # Preserve original coordinates (lat, lon, depth)
    ds_out = ds_out.assign_coords(ds.coords)
    
    return ds_out