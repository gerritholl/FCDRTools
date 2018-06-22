import unittest

import xarray as xr

from fiduceo.cdr.writer.templates.albedo import Albedo


class AlbedoTest(unittest.TestCase):

    def test_add_variables(self):
        ds = xr.Dataset()
        Albedo.add_variables(ds)