import unittest

import xarray as xr

from fiduceo.cdr.writer.templates.albedo import Albedo


class AlbedoTest(unittest.TestCase):

    def test_add_variables(self):
        ds = xr.Dataset()
        Albedo.add_variables(ds)
        # @todo 1 tb/tb continue here when geolocation question os resolved 2018-06-25