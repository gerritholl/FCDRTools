import unittest

import numpy as np
import xarray as xr

from fiduceo.cdr.writer.templates.uth import UTH
from fiduceo.common.test.assertions import Assertions


class UTHTest(unittest.TestCase):

    def test_add_variables(self):
        ds = xr.Dataset()

        UTH.add_variables(ds, 360, 100)

        Assertions.assert_gridded_geolocation_variables(self, ds, 360, 100)
        Assertions.assert_quality_flags(self, ds, 360, 100)

        time_ranges = ds.variables["time_ranges"]
        self.assertEqual((2, 100, 360), time_ranges.shape)
        self.assertEqual(-1, time_ranges.values[0, 66, 178])
        self.assertEqual(4294967295, time_ranges.attrs["_FillValue"])
        self.assertEqual("Minimum and maximum seconds of day pixel contribution time", time_ranges.attrs["description"])
        self.assertEqual("s", time_ranges.attrs["units"])
        self.assertEqual("lon lat", time_ranges.attrs["coordinates"])

        obs_count = ds.variables["observation_count"]
        self.assertEqual((100, 360), obs_count.shape)
        self.assertEqual(-32767, obs_count.values[67, 179])
        self.assertEqual(-32767, obs_count.attrs["_FillValue"])
        self.assertEqual("Number of observations contributing to pixel value", obs_count.attrs["description"])
        self.assertEqual("lon lat", obs_count.attrs["coordinates"])

        uth = ds.variables["uth"]
        self.assertEqual((100, 360), uth.shape)
        self.assertTrue(np.isnan(uth.values[97, 198]))
        self.assertTrue(np.isnan(uth.attrs["_FillValue"]))
        self.assertEqual("lon lat", uth.attrs["coordinates"])

        u_ind_uth = ds.variables["u_independent_uth"]
        self.assertEqual((100, 360), u_ind_uth.shape)
        self.assertTrue(np.isnan(u_ind_uth.values[98, 199]))
        self.assertTrue(np.isnan(u_ind_uth.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to independent effects", u_ind_uth.attrs["description"])
        self.assertEqual("lon lat", u_ind_uth.attrs["coordinates"])

        u_str_uth = ds.variables["u_structured_uth"]
        self.assertEqual((100, 360), u_str_uth.shape)
        self.assertTrue(np.isnan(u_str_uth.values[0, 201]))
        self.assertTrue(np.isnan(u_str_uth.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to structured effects", u_str_uth.attrs["description"])
        self.assertEqual("lon lat", u_str_uth.attrs["coordinates"])

        u_com_uth = ds.variables["u_common_uth"]
        self.assertEqual((100, 360), u_com_uth.shape)
        self.assertTrue(np.isnan(u_com_uth.values[70, 171]))
        self.assertTrue(np.isnan(u_com_uth.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to common effects", u_com_uth.attrs["description"])
        self.assertEqual("lon lat", u_com_uth.attrs["coordinates"])
