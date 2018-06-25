import unittest

import xarray as xr

from fiduceo.cdr.writer.templates.aot import AOT
from fiduceo.common.test.assertions import Assertions


class AOTTest(unittest.TestCase):

    CHUNKING = (1280, 409)

    def test_add_variables(self):
        ds = xr.Dataset()

        AOT.add_variables(ds, 409, 12876)

        Assertions.assert_geolocation_variables(self, ds, 409, 12876, chunking=self.CHUNKING)
        Assertions.assert_quality_flags(self, ds, 409, 12876, chunking=self.CHUNKING)
