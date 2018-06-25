import unittest

import xarray as xr

from fiduceo.cdr.writer.templates.uth import UTH
from fiduceo.common.test.assertions import Assertions


class UTHTest(unittest.TestCase):

    def test_add_variables(self):
        ds = xr.Dataset()

        UTH.add_variables(ds, 360, 100)

        Assertions.assert_gridded_geolocation_variables(self, ds, 360, 100)
        Assertions.assert_quality_flags(self, ds, 360, 100)