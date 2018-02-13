import unittest

import xarray as xr

from fiduceo.fcdr.test.writer.templates.assertions import Assertions
from fiduceo.fcdr.test.writer.templates.hirs_assert import HIRSAssert
from fiduceo.fcdr.writer.templates.hirs_4 import HIRS4

CHUNKING_3D = (10, 512, 56)
CHUNKING_2D = (512, 56)


class HIRS4Test(unittest.TestCase):
    def test_add_original_variables(self):
        ha = HIRSAssert()
        ds = xr.Dataset()
        HIRS4.add_original_variables(ds, 6)

        Assertions.assert_geolocation_variables(self, ds, 56, 6, chunking=CHUNKING_2D)
        Assertions.assert_quality_flags(self, ds, 56, 6, chunking=CHUNKING_2D)

        ha.assert_bt_variable(ds, chunking=CHUNKING_3D)
        ha.assert_common_angles(ds, chunking=CHUNKING_2D)
        ha.assert_common_sensor_variables(ds)
        ha.assert_extended_quality_flags(ds)

    def test_get_swath_width(self):
        self.assertEqual(56, HIRS4.get_swath_width())

    def test_add_easy_fcdr_variables(self):
        ha = HIRSAssert()
        ds = xr.Dataset()
        HIRS4.add_easy_fcdr_variables(ds, 7)

        ha.assert_easy_fcdr_uncertainties(ds, chunking=CHUNKING_3D)

    def test_add_full_fcdr_variables(self):
        ha = HIRSAssert()
        ds = xr.Dataset()
        HIRS4.add_full_fcdr_variables(ds, 6)
        ha.assert_minor_frame_flags(ds)
