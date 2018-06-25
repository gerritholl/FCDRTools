import unittest

import xarray as xr

from fiduceo.cdr.writer.templates.sst import SST
from fiduceo.common.test.assertions import Assertions


class SSTTest(unittest.TestCase):
    CHUNKING = (1280, 409)

    def test_add_variables(self):
        ds = xr.Dataset()

        SST.add_variables(ds, 409, 12877)

        Assertions.assert_geolocation_variables(self, ds, 409, 12877, chunking=self.CHUNKING)
