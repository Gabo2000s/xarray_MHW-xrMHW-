"""unittest-style mirror of the core cases in ``test_xrMHW.py``.

Kept so the suite can be run with ``python -m unittest`` in environments
without pytest. The authoritative suite is ``test_xrMHW.py`` plus
``test_against_marineHeatWaves.py``; if you add a case, add it there.
"""

import unittest

import numpy as np
import pandas as pd
import xarray as xr

import xrMHW

class TestXrMHW(unittest.TestCase):
    def create_synthetic_dataset(self, temp_values):
        """Helper to create a synthetic 3D dataset for a single spatial point."""
        T = len(temp_values)
        times = pd.date_range("1990-01-01", periods=T)

        da = xr.DataArray(
            temp_values[:, np.newaxis, np.newaxis],
            coords=[times, [0], [0]],
            dims=["time", "lat", "lon"],
        )
        return xr.Dataset({"thetao": da})

    def test_mhw_detection_basic(self):
        """Verify that a basic 6-day temperature spike is correctly identified as a Marine Heatwave."""
        T = 365 * 2
        temp = np.full(T, 10.0)
        temp[400:406] = 15.0
        ds = self.create_synthetic_dataset(temp)
        res = xrMHW.xrMHW_func(
            ds, 'thetao', (1990, 1990),
            min_duration=5, max_gap=2, join_across_gaps=True
        )

        self.assertIn('mhw_intensity', res)
        self.assertIn('mhw_severity', res)
        self.assertIn('mhw_duration', res)
        self.assertIn('mhw_threshold', res)

        is_mhw = ~np.isnan(res['mhw_intensity'].values[0, 0, :])
        self.assertTrue(is_mhw[400:406].all())
        self.assertFalse(is_mhw[:400].any())
        self.assertFalse(is_mhw[406:].any())
        self.assertEqual(res['mhw_duration'].values[0, 0, 400], 6)

    def test_gap_joining_methods(self):
        """Verify that pre-filter joins the events while post-filter discards short individual spikes."""
        T = 730
        temp = np.full(T, 10.0)
        temp[400:404] = 15.0  # Spike 1 (4 days)
        temp[406:410] = 15.0  # Spike 2 (4 days)
        ds = self.create_synthetic_dataset(temp)

        res_post = xrMHW.xrMHW_func(
            ds, 'thetao', (1990, 1990),
            min_duration=5, max_gap=2, join_across_gaps=True, join_method='post-filter'
        )
        is_mhw_post = ~np.isnan(res_post['mhw_intensity'].values[0, 0, :])
        self.assertFalse(is_mhw_post.any())

        res_pre = xrMHW.xrMHW_func(
            ds, 'thetao', (1990, 1990),
            min_duration=5, max_gap=2, join_across_gaps=True, join_method='pre-filter'
        )
        is_mhw_pre = ~np.isnan(res_pre['mhw_intensity'].values[0, 0, :])
        self.assertTrue(is_mhw_pre[400:410].all())
        self.assertEqual(res_pre['mhw_duration'].values[0, 0, 400], 10)

    def test_nan_land_mask(self):
        """Verify that land mask coordinates (all NaNs) are skipped and return NaNs immediately."""
        temp = np.full(365, np.nan)
        ds = self.create_synthetic_dataset(temp)
        res = xrMHW.xrMHW_func(ds, 'thetao', (1990, 1990))

        self.assertTrue(np.isnan(res['mhw_intensity'].values).all())
        self.assertTrue(np.isnan(res['climatology'].values).all())
        self.assertTrue(np.isnan(res['mhw_threshold'].values).all())

    def test_cold_spells(self):
        """Verify that Marine Cold Spells (MCS) are correctly detected when cold_spells=True."""
        T = 365 * 2
        temp = np.full(T, 20.0)
        temp[400:406] = 5.0
        ds = self.create_synthetic_dataset(temp)
        res = xrMHW.xrMHW_func(ds, 'thetao', (1990, 1990),
                               cold_spells=True, min_duration=5)

        self.assertIn('mcs_intensity', res)
        self.assertIn('mcs_duration', res)
        self.assertIn('mcs_threshold', res)

        is_mcs = ~np.isnan(res['mcs_intensity'].values[0, 0, :])
        self.assertTrue(is_mcs[400:406].all())
        self.assertFalse(is_mcs[:400].any())
        self.assertFalse(is_mcs[406:].any())


if __name__ == '__main__':
    unittest.main()
