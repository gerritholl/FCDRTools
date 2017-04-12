import unittest

import xarray as xr

from writer.templates.templateutil import TemplateUtil


class TemplateUtilTest(unittest.TestCase):
    def test_add_golocation_variables(self):
        ds = xr.Dataset()
        TemplateUtil.add_geolocation_variables(ds, 8, 10)

        latitude = ds.variables["latitude"]
        self.assertEqual((10, 8), latitude.shape)
        self.assertEqual(-32768, latitude.data[4, 5])
        self.assertEqual(-32768, latitude.attrs["_FillValue"])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])
        self.assertEqual(0.0027466658, latitude.attrs["scale_factor"])

        longitude = ds.variables["longitude"]
        self.assertEqual((10, 8), longitude.shape)
        self.assertEqual(-32768, longitude.data[5, 6])
        self.assertEqual(-32768, longitude.attrs["_FillValue"])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])
        self.assertEqual(0.0054933317, longitude.attrs["scale_factor"])
