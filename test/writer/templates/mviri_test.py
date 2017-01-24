import unittest

import numpy as np
import xarray as xr

from writer.default_data import DefaultData
from writer.templates.mviri import MVIRI


class MVIRITest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        MVIRI.add_original_variables(ds, 5)

        latitude = ds.variables["latitude"]
        self.assertEqual((5, 4000), latitude.shape)
        self.assertEqual(-32768.0, latitude.data[1, 106])
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])

        longitude = ds.variables["longitude"]
        self.assertEqual((5, 4000), longitude.shape)
        self.assertEqual(-32768.0, longitude.data[1, 107])
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])

        time = ds.variables["time"]
        self.assertEqual((5,), time.shape)
        self.assertEqual(-2147483647, time.data[4])
        self.assertEqual(-2147483647, time.attrs["_FillValue"])
        self.assertEqual("time", time.attrs["standard_name"])
        self.assertEqual("Acquisition time in seconds since 1970-01-01 00:00:00", time.attrs["long_name"])
        self.assertEqual("s", time.attrs["units"])

        time_delta = ds.variables["time_delta"]
        self.assertEqual((5, 4000), time_delta.shape)
        self.assertEqual(-127, time_delta.data[2, 108])
        self.assertEqual(-127, time_delta.attrs["_FillValue"])
        self.assertEqual("time", time_delta.attrs["standard_name"])
        self.assertEqual("Acquisition time delta", time_delta.attrs["long_name"])
        self.assertEqual("s", time_delta.attrs["units"])
        self.assertEqual(0.025, time_delta.attrs["scale_factor"])

        sat_azimuth = ds.variables["satellite_azimuth_angle"]
        self.assertEqual((5, 4000), sat_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sat_azimuth.data[0, 109])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sat_azimuth.attrs["_FillValue"])
        self.assertEqual("sensor_azimuth_angle", sat_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sat_azimuth.attrs["units"])

        sat_zenith = ds.variables["satellite_zenith_angle"]
        self.assertEqual((5, 4000), sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sat_zenith.data[0, 110])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sat_zenith.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_zenith.attrs["standard_name"])
        self.assertEqual("degree", sat_zenith.attrs["units"])

        sol_azimuth = ds.variables["solar_azimuth_angle"]
        self.assertEqual((5, 4000), sol_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_azimuth.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_azimuth.attrs["_FillValue"])
        self.assertEqual("solar_azimuth_angle", sol_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sol_azimuth.attrs["units"])

        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertEqual((5, 4000), sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_zenith.data[0, 112])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_zenith.attrs["_FillValue"])
        self.assertEqual("solar_zenith_angle", sol_zenith.attrs["standard_name"])
        self.assertEqual("degree", sol_zenith.attrs["units"])

        count = ds.variables["count"]
        self.assertEqual((5, 4000), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), count.data[0, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), count.attrs["_FillValue"])
        self.assertEqual("Image counts", count.attrs["standard_name"])
        self.assertEqual("count", count.attrs["units"])