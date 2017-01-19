import unittest

import xarray as xr

from writer.templates.mviri import MVIRI


class MVIRITest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        MVIRI.add_original_variables(ds, 3, 4)

        latitude = ds.variables["latitude"]
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])
        self.assertEqual(-32768.0, latitude.data[0, 0])

        longitude = ds.variables["longitude"]
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])
        self.assertEqual(-32768.0, longitude.data[0, 0])
