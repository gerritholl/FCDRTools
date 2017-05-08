import unittest

import numpy as np

from fiduceo.fcdr.writer.default_data import DefaultData


class HIRSAssert(unittest.TestCase):
    def assert_geolocation(self, ds):
        latitude = ds.variables["latitude"]
        self.assertEqual((6, 56), latitude.shape)
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual(-32768.0, latitude.data[0, 0])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])

        longitude = ds.variables["longitude"]
        self.assertEqual((6, 56), latitude.shape)
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual(-32768.0, longitude.data[0, 0])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])

    def assert_bt_variable(self, ds):
        bt = ds.variables["bt"]
        self.assertEqual((19, 6, 56), bt.shape)
        self.assertEqual(-999, bt.data[0, 2, 1])
        self.assertEqual(-999, bt.attrs["_FillValue"])
        self.assertEqual("toa_brightness_temperature", bt.attrs["standard_name"])
        self.assertEqual("Brightness temperature, NOAA/EUMETSAT calibrated", bt.attrs["long_name"])
        self.assertEqual("K", bt.attrs["units"])
        self.assertEqual(0.01, bt.attrs["scale_factor"])
        self.assertEqual(150, bt.attrs["add_offset"])
        self.assertEqual("scnlinf scantype qualind linqualflags chqualflags mnfrqualflags", bt.attrs["ancilliary_variables"])

    def assert_common_angles(self, ds):
        satellite_zenith_angle = ds.variables["satellite_zenith_angle"]
        self.assertEqual((6, 56), satellite_zenith_angle.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), satellite_zenith_angle.data[2, 2])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), satellite_zenith_angle.attrs["_FillValue"])
        self.assertEqual(0.01, satellite_zenith_angle.attrs["scale_factor"])
        self.assertEqual(-180.0, satellite_zenith_angle.attrs["add_offset"])
        self.assertEqual("platform_zenith_angle", satellite_zenith_angle.attrs["standard_name"])
        self.assertEqual("degree", satellite_zenith_angle.attrs["units"])

        solar_zenith_angle = ds.variables["solar_zenith_angle"]
        self.assertEqual((6, 56), solar_zenith_angle.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), solar_zenith_angle.data[3, 3])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), solar_zenith_angle.attrs["_FillValue"])
        self.assertEqual(0.01, solar_zenith_angle.attrs["scale_factor"])
        self.assertEqual(-180.0, solar_zenith_angle.attrs["add_offset"])
        self.assertEqual("solar_zenith_angle", solar_zenith_angle.attrs["standard_name"])
        self.assertEqual("solar_zenith_angle", solar_zenith_angle.attrs["orig_name"])
        self.assertEqual("degree", solar_zenith_angle.attrs["units"])

        satellite_azimuth_angle = ds.variables["satellite_azimuth_angle"]
        self.assertEqual((6, 56), satellite_azimuth_angle.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), satellite_azimuth_angle.data[5, 5])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), satellite_azimuth_angle.attrs["_FillValue"])
        self.assertEqual(0.01, satellite_azimuth_angle.attrs["scale_factor"])
        self.assertEqual(-180.0, satellite_azimuth_angle.attrs["add_offset"])
        self.assertEqual("sensor_azimuth_angle", satellite_azimuth_angle.attrs["standard_name"])
        self.assertEqual("local_azimuth_angle", satellite_azimuth_angle.attrs["long_name"])
        self.assertEqual("degree", satellite_azimuth_angle.attrs["units"])

        solar_azimuth_angle = ds.variables["solar_azimuth_angle"]
        self.assertEqual((6, 56), solar_azimuth_angle.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), solar_azimuth_angle.data[4, 4])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), solar_azimuth_angle.attrs["_FillValue"])
        self.assertEqual(0.01, solar_azimuth_angle.attrs["scale_factor"])
        self.assertEqual(-180.0, solar_azimuth_angle.attrs["add_offset"])
        self.assertEqual("solar_azimuth_angle", solar_azimuth_angle.attrs["standard_name"])
        self.assertEqual("degree", solar_azimuth_angle.attrs["units"])
