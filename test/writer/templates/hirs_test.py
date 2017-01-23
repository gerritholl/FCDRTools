import unittest

import numpy as np
import xarray as xr

from writer.default_data import DefaultData
from writer.templates.hirs import HIRS


class HIRSTest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        HIRS.add_original_variables(ds, 6)

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

        bt = ds.variables["bt"]
        self.assertEqual((19, 6, 56), bt.shape)
        self.assertEqual(-999.0, bt.data[0, 2, 1])
        self.assertEqual(-999.0, bt.attrs["_FillValue"])
        self.assertEqual("toa_brightness_temperature", bt.attrs["standard_name"])
        self.assertEqual("K", bt.attrs["units"])

        c_earth = ds.variables["c_earth"]
        self.assertEqual((20, 6, 56), c_earth.shape)
        self.assertEqual(99999, c_earth.data[0, 2, 3])
        self.assertEqual(99999, c_earth.attrs["_FillValue"])
        self.assertEqual("counts_Earth", c_earth.attrs["standard_name"])
        self.assertEqual("count", c_earth.attrs["units"])

        l_earth = ds.variables["L_earth"]
        self.assertEqual((20, 6, 56), l_earth.shape)
        self.assertEqual(-999.0, l_earth.data[0, 2,4])
        self.assertEqual(-999.0, l_earth.attrs["_FillValue"])
        self.assertEqual("toa_outgoing_inband_radiance", l_earth.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", l_earth.attrs["units"])

        sat_za = ds.variables["sat_za"]
        self.assertEqual((6, 56), sat_za.shape)
        self.assertEqual(-999.0, sat_za.data[2, 2])
        self.assertEqual(-999.0, sat_za.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_za.attrs["standard_name"])
        self.assertEqual("degree", sat_za.attrs["units"])

        scanline = ds.variables["scanline"]
        self.assertEqual((6,), scanline.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.data[3])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.attrs["_FillValue"])
        self.assertEqual("scanline_number", scanline.attrs["standard_name"])
        self.assertEqual("number", scanline.attrs["units"])

        scnlinf = ds.variables["scnlinf"]
        self.assertEqual((6,), scnlinf.shape)
        self.assertEqual(9, scnlinf.data[4])
        self.assertEqual(9, scnlinf.attrs["_FillValue"])
        self.assertEqual("0, 1, 2, 3", scnlinf.attrs["flag_values"])
        self.assertEqual("earth_view space_view icct_view iwct_view", scnlinf.attrs["flag_meanings"])
        self.assertEqual("scanline_bitfield", scnlinf.attrs["standard_name"])
