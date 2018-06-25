import unittest

import xarray as xr

from fiduceo.cdr.writer.templates.albedo import Albedo
from fiduceo.common.test.assertions import Assertions


class AlbedoTest(unittest.TestCase):

    CHUNKING = (500, 500)

    def test_add_variables(self):
        ds = xr.Dataset()
        Albedo.add_variables(ds, 5000, 5000)

        # @todo 1 tb/tb continue here when geolocation question is resolved 2018-06-25

        Assertions.assert_quality_flags(self, ds, 5000, 5000, chunking=self.CHUNKING)