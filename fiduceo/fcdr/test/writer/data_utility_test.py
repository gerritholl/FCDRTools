import unittest

import numpy as np
from xarray import Variable

from fiduceo.fcdr.writer.default_data import DefaultData


class DataUtilityTest(unittest.TestCase):
    def test_check_scaling_ranges_int8_vector_ok(self):
        default_array = DefaultData.create_default_vector(4, np.int8)
        variable = Variable(["y"], default_array)

        iis = np.iinfo(np.int8)
        print(iis.min)
        print(iis.max)
