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

        time_ranges_asc = ds.variables["time_ranges_ascending"]
        self.assertEqual((2, 100, 360), time_ranges_asc.shape)
        self.assertEqual(-1, time_ranges_asc.values[0, 66, 178])
        self.assertEqual(4294967295, time_ranges_asc.attrs["_FillValue"])
        self.assertEqual("Minimum and maximum seconds of day pixel contribution time, ascending nodes", time_ranges_asc.attrs["description"])
        self.assertEqual("s", time_ranges_asc.attrs["units"])
        self.assertEqual("lon lat", time_ranges_asc.attrs["coordinates"])

        time_ranges_desc = ds.variables["time_ranges_descending"]
        self.assertEqual((2, 100, 360), time_ranges_desc.shape)
        self.assertEqual(-1, time_ranges_desc.values[1, 67, 179])
        self.assertEqual(4294967295, time_ranges_desc.attrs["_FillValue"])
        self.assertEqual("Minimum and maximum seconds of day pixel contribution time, descending nodes", time_ranges_desc.attrs["description"])
        self.assertEqual("s", time_ranges_desc.attrs["units"])
        self.assertEqual("lon lat", time_ranges_desc.attrs["coordinates"])

        obs_count_asc = ds.variables["observation_count_ascending"]
        self.assertEqual((100, 360), obs_count_asc.shape)
        self.assertEqual(-32767, obs_count_asc.values[67, 179])
        self.assertEqual(-32767, obs_count_asc.attrs["_FillValue"])
        self.assertEqual("Number of observations contributing to pixel value, ascending nodes", obs_count_asc.attrs["description"])
        self.assertEqual("lon lat", obs_count_asc.attrs["coordinates"])

        obs_count_desc = ds.variables["observation_count_descending"]
        self.assertEqual((100, 360), obs_count_desc.shape)
        self.assertEqual(-32767, obs_count_desc.values[68, 180])
        self.assertEqual(-32767, obs_count_desc.attrs["_FillValue"])
        self.assertEqual("Number of observations contributing to pixel value, descending nodes", obs_count_desc.attrs["description"])
        self.assertEqual("lon lat", obs_count_desc.attrs["coordinates"])

        uth_asc = ds.variables["uth_ascending"]
        self.assertEqual((100, 360), uth_asc.shape)
        self.assertTrue(np.isnan(uth_asc.values[97, 198]))
        self.assertTrue(np.isnan(uth_asc.attrs["_FillValue"]))
        self.assertEqual("lon lat", uth_asc.attrs["coordinates"])

        uth_desc = ds.variables["uth_descending"]
        self.assertEqual((100, 360), uth_desc.shape)
        self.assertTrue(np.isnan(uth_desc.values[98, 199]))
        self.assertTrue(np.isnan(uth_desc.attrs["_FillValue"]))
        self.assertEqual("lon lat", uth_desc.attrs["coordinates"])

        u_ind_uth_asc = ds.variables["u_independent_uth_ascending"]
        self.assertEqual((100, 360), u_ind_uth_asc.shape)
        self.assertTrue(np.isnan(u_ind_uth_asc.values[98, 199]))
        self.assertTrue(np.isnan(u_ind_uth_asc.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to independent effects, ascending nodes", u_ind_uth_asc.attrs["description"])
        self.assertEqual("lon lat", u_ind_uth_asc.attrs["coordinates"])

        u_ind_uth_desc = ds.variables["u_independent_uth_descending"]
        self.assertEqual((100, 360), u_ind_uth_desc.shape)
        self.assertTrue(np.isnan(u_ind_uth_desc.values[99, 200]))
        self.assertTrue(np.isnan(u_ind_uth_desc.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to independent effects, descending nodes", u_ind_uth_desc.attrs["description"])
        self.assertEqual("lon lat", u_ind_uth_desc.attrs["coordinates"])

        u_str_uth_asc = ds.variables["u_structured_uth_ascending"]
        self.assertEqual((100, 360), u_str_uth_asc.shape)
        self.assertTrue(np.isnan(u_str_uth_asc.values[0, 201]))
        self.assertTrue(np.isnan(u_str_uth_asc.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to structured effects, ascending nodes", u_str_uth_asc.attrs["description"])
        self.assertEqual("lon lat", u_str_uth_asc.attrs["coordinates"])

        u_str_uth_desc = ds.variables["u_structured_uth_descending"]
        self.assertEqual((100, 360), u_str_uth_desc.shape)
        self.assertTrue(np.isnan(u_str_uth_desc.values[1, 202]))
        self.assertTrue(np.isnan(u_str_uth_desc.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to structured effects, descending nodes", u_str_uth_desc.attrs["description"])
        self.assertEqual("lon lat", u_str_uth_desc.attrs["coordinates"])

        u_com_uth_asc = ds.variables["u_common_uth_ascending"]
        self.assertEqual((100, 360), u_com_uth_asc.shape)
        self.assertTrue(np.isnan(u_com_uth_asc.values[70, 171]))
        self.assertTrue(np.isnan(u_com_uth_asc.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to common effects, ascending nodes", u_com_uth_asc.attrs["description"])
        self.assertEqual("lon lat", u_com_uth_asc.attrs["coordinates"])

        u_com_uth_desc = ds.variables["u_common_uth_descending"]
        self.assertEqual((100, 360), u_com_uth_desc.shape)
        self.assertTrue(np.isnan(u_com_uth_desc.values[71, 172]))
        self.assertTrue(np.isnan(u_com_uth_desc.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of uth due to common effects, descending nodes", u_com_uth_desc.attrs["description"])
        self.assertEqual("lon lat", u_com_uth_desc.attrs["coordinates"])
