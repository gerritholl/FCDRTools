import unittest

import numpy as np
import xarray as xr

from fiduceo.fcdr.test.writer.templates.hirs_assert import HIRSAssert
from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.hirs_2 import HIRS2


class HIRS2Test(unittest.TestCase):
    def test_add_original_variables(self):
        ha = HIRSAssert()
        ds = xr.Dataset()
        HIRS2.add_original_variables(ds, 6)

        ha.assert_geolocation(ds)
        ha.assert_bt_variable(ds)
        self._assert_angle_variables(ds)
        ha.assert_common_sensor_variables(ds)

    def test_get_swath_width(self):
        self.assertEqual(56, HIRS2.get_swath_width())

    def test_add_easy_fcdr_variables(self):
        ha = HIRSAssert()
        ds = xr.Dataset()
        HIRS2.add_easy_fcdr_variables(ds, 7)

        ha.assert_easy_fcdr_uncertainties(ds)

    def test_add_full_fcdr_variables(self):
        pass

    def _assert_angle_variables(self, ds):
        satellite_zenith_angle = ds.variables["satellite_zenith_angle"]
        self.assertEqual((6,), satellite_zenith_angle.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), satellite_zenith_angle.data[3])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), satellite_zenith_angle.attrs["_FillValue"])
        self.assertEqual(0.01, satellite_zenith_angle.attrs["scale_factor"])
        self.assertEqual(-180.0, satellite_zenith_angle.attrs["add_offset"])
        self.assertEqual("platform_zenith_angle", satellite_zenith_angle.attrs["standard_name"])
        self.assertEqual("degree", satellite_zenith_angle.attrs["units"])

        solar_azimuth_angle = ds.variables["solar_azimuth_angle"]
        self.assertEqual((6, 56), solar_azimuth_angle.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), solar_azimuth_angle.data[4, 4])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), solar_azimuth_angle.attrs["_FillValue"])
        self.assertEqual(0.01, solar_azimuth_angle.attrs["scale_factor"])
        self.assertEqual(-180.0, solar_azimuth_angle.attrs["add_offset"])
        self.assertEqual("solar_azimuth_angle", solar_azimuth_angle.attrs["standard_name"])
        self.assertEqual("degree", solar_azimuth_angle.attrs["units"])
