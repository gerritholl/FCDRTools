import unittest

import xarray as xr

from fiduceo.fcdr.writer.templates.hirs_3 import HIRS3
from fiduceo.fcdr.test.writer.templates.hirs_assert import HIRSAssert


class HIRS3Test(unittest.TestCase):
    def test_add_original_variables(self):
        ha = HIRSAssert()
        ds = xr.Dataset()
        HIRS3.add_original_variables(ds, 6)

        ha.assert_geolocation(ds)
        ha.assert_bt_variable(ds)
        ha.assert_common_angles(ds)

    def test_get_swath_width(self):
        pass

    def test_add_easy_fcdr_variables(self):
        pass

    def test_add_full_fcdr_variables(self):
        pass