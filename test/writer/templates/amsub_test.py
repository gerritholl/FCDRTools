import unittest

import xarray as xr

from writer.templates.amsub import AMSUB


class AMSUBTest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        AMSUB.add_original_variables(ds, 2, 5)

        latitude = ds.variables["latitude"]
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])
        self.assertEqual((2, 5), latitude.shape)
        self.assertEqual(-32768.0, latitude.data[0, 0])

        longitude = ds.variables["longitude"]
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])
        self.assertEqual((2, 5), longitude.shape)
        self.assertEqual(-32768.0, longitude.data[1, 0])

# @todo 1 tb/tb continue here 2017-01-19
        #btemps = ds.variables["btemps"]
