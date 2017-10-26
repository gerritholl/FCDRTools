import unittest

import xarray as xr

from fiduceo.fcdr.test.writer.templates.assertions import Assertions
from fiduceo.fcdr.writer.templates.hirs_3 import HIRS3
from fiduceo.fcdr.test.writer.templates.hirs_assert import HIRSAssert


class HIRS3Test(unittest.TestCase):
    def test_add_original_variables(self):
        ha = HIRSAssert()
        ds = xr.Dataset()
        HIRS3.add_original_variables(ds, 6)

        Assertions.assert_geolocation_variables(self, ds, 56, 6)
        Assertions.assert_quality_flags(self, ds, 56, 6)

        ha.assert_bt_variable(ds)
        ha.assert_common_angles(ds)
        ha.assert_common_sensor_variables(ds)
        ha.assert_extended_quality_flags(ds)

    def test_get_swath_width(self):
        self.assertEqual(56, HIRS3.get_swath_width())

    def test_add_easy_fcdr_variables(self):
        ha = HIRSAssert()
        ds = xr.Dataset()
        HIRS3.add_easy_fcdr_variables(ds, 7)

        ha.assert_easy_fcdr_uncertainties(ds)

    def test_add_full_fcdr_variables(self):
        pass