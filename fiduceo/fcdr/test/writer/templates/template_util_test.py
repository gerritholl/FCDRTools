import unittest

import numpy as np
import xarray as xr

from fiduceo.fcdr.writer.templates.templateutil import TemplateUtil


class TemplateUtilTest(unittest.TestCase):
    def test_add_golocation_variables(self):
        ds = xr.Dataset()
        TemplateUtil.add_geolocation_variables(ds, 8, 10)

        latitude = ds.variables["latitude"]
        self.assertEqual((10, 8), latitude.shape)
        self.assertTrue(np.isnan(latitude.data[4, 4]))
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])
        self.assertEqual(np.int16, latitude.encoding['dtype'])
        self.assertEqual(-32768, latitude.encoding['_FillValue'])
        self.assertEqual(0.0027466658, latitude.encoding['scale_factor'])
        self.assertEqual(0.0, latitude.encoding['add_offset'])

        longitude = ds.variables["longitude"]
        self.assertEqual((10, 8), longitude.shape)
        self.assertTrue(np.isnan(longitude.data[5, 6]))
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])
        self.assertEqual(np.int16, longitude.encoding['dtype'])
        self.assertEqual(-32768, longitude.encoding['_FillValue'])
        self.assertEqual(0.0054933317, longitude.encoding['scale_factor'])
        self.assertEqual(0.0, longitude.encoding['add_offset'])
