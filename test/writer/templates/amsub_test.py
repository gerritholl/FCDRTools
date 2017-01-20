import unittest

import xarray as xr

from writer.templates.amsub import AMSUB


class AMSUBTest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        AMSUB.add_original_variables(ds, 2, 4)

        latitude = ds.variables["latitude"]
        self.assertEqual((4, 2), latitude.shape)
        self.assertEqual(-32768.0, latitude.data[0, 0])
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])

        longitude = ds.variables["longitude"]
        self.assertEqual((4, 2), longitude.shape)
        self.assertEqual(-32768.0, longitude.data[1, 0])
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])

        btemps = ds.variables["btemps"]
        self.assertEqual((5, 4, 2), btemps.shape)
        self.assertEqual(-999999, btemps.data[0, 2, 0])
        self.assertEqual(-999999, btemps.attrs["_FillValue"])
        self.assertEqual("toa_brightness_temperature", btemps.attrs["standard_name"])
        self.assertEqual("K", btemps.attrs["units"])
        self.assertEqual(0.01, btemps.attrs["scale_factor"])
