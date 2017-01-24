import unittest

import numpy as np
import xarray as xr

from writer.default_data import DefaultData
from writer.templates.easy_fcdr import EasyFCDR


class EasyFCDRTest(unittest.TestCase):
    def test_add_variables(self):
        ds = xr.Dataset()

        EasyFCDR.add_variables(ds, 10, 12)

        u_random = ds.variables["u_random"]
        self.assertEqual((12, 10), u_random.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_random.data[3, 2])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_random.attrs["_FillValue"])
        self.assertEqual("random uncertainty", u_random.attrs["standard_name"])

        u_systematic = ds.variables["u_systematic"]
        self.assertEqual((12, 10), u_systematic.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_systematic.data[3, 2])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_systematic.attrs["_FillValue"])
        self.assertEqual("systematic uncertainty", u_systematic.attrs["standard_name"])