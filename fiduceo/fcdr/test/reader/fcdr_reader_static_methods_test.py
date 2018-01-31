import unittest as ut
import xarray as xr
import numpy as np

import fiduceo.fcdr.test.test_utils as ftu
from fiduceo.fcdr.reader.fcdr_reader import FCDRReader as RD


class FCDRReaderStaticMethodsTest(ut.TestCase):

    def setUp(self):
        ds = xr.Dataset()
        ds['asd'] = xr.Variable(['z', 'y', 'x'], np.full([3, 4, 5], 13))
        ds['lsmf'] = xr.Variable(['y', 'x'], np.full([4, 5], 14))
        ds['frasdsn'] = xr.Variable(['y', 'x'], np.full([4, 5], 15))
        ds['ka'] = xr.Variable(['x'], np.full([5], 16))
        ds['ody'] = xr.Variable(['y'], range(17, 21))
        ds['woody'] = xr.Variable(['y'], np.full([4], 18))
        self.ds = ds
        self.dic = RD._create_dictionary_of_non_virtuals(ds)

    def test_GetBiggestDimension(self):
        expression = '(asd)*[lsmf + frasdsn] - ka'
        biggest_variable = RD._get_biggest_variable(self.dic, expression)
        self.assertEqual(("z", "y", "x"), biggest_variable.dims)

        expression = '[lsmf + frasdsn] - ka'
        biggest_variable = RD._get_biggest_variable(self.dic, expression)
        self.assertEqual(("y", "x"), biggest_variable.dims)

        expression = 'lsmf +  ka'
        biggest_variable = RD._get_biggest_variable(self.dic, expression)
        self.assertEqual(("y", "x"), biggest_variable.dims)

        expression = 'frasdsn - ka'
        biggest_variable = RD._get_biggest_variable(self.dic, expression)
        self.assertEqual(("y", "x"), biggest_variable.dims)

        expression = 'ka * 4'
        biggest_variable = RD._get_biggest_variable(self.dic, expression)
        self.assertEqual(("x",), biggest_variable.dims)

    def test_GetKeysSortesd_LongestFirst(self):
        dictonary = {"a": 1, "a2": 2, "b": 3, "bb3": 4, "c2": 5}
        sorted_keys = RD._get_keys_sorted__longest_first(dictonary)
        self.assertEqual(["bb3", "a2", "c2", "a", "b"], sorted_keys)

    def test_ExpandOneDimensionalVariables(self):
        pass

    def test_FindUsedOneDimensionalVerticalToExtend_emptyResult(self):
        """
        'ka' is a one dimensional variable but not y dimension.
        Only variables with 'y' dimension must be extended.
        """
        expression = 'asd * ka'
        biggest_variable = RD._get_biggest_variable(self.dic, expression)
        dims = biggest_variable.dims
        one_d_vars = RD._find_used_one_dimensional_variables_to_extend(self.dic, dims, expression)
        self.assertEquals(list(), one_d_vars)

    def test_FindUsedOneDimensionalVerticalToExtend_oneResult(self):
        """
        'ody' is a one dimensional variable with dimension 'y'
        """
        expression = 'asd * ody + ka'
        biggest_variable = RD._get_biggest_variable(self.dic, expression)
        dims = biggest_variable.dims
        one_d_vars = RD._find_used_one_dimensional_variables_to_extend(self.dic, dims, expression)
        self.assertEquals(1, len(one_d_vars))
        self.assertTrue('ody' in one_d_vars)
        variable = self.dic['ody']
        self.assertEqual(('y',), variable.dims)
        self.assertEqual(17, variable.data[0])

    def test_FindUsedOneDimensionalVerticalToExtend_twoResults(self):
        """
        'ody' and 'woody' are a one dimensional variables with dimension 'y'
        """
        expression = 'asd * (ody - woody) + ka'
        biggest_variable = RD._get_biggest_variable(self.dic, expression)
        dims = biggest_variable.dims
        one_d_vars = RD._find_used_one_dimensional_variables_to_extend(self.dic, dims, expression)
        self.assertEquals(2, len(one_d_vars))
        self.assertTrue('ody' in one_d_vars)
        self.assertTrue('woody' in one_d_vars)
        var_o = self.dic['ody']
        self.assertEqual(('y',), var_o.dims)
        self.assertEqual(17, var_o.data[0])
        var_w = self.dic['woody']
        self.assertEqual(('y',), var_w.dims)
        self.assertEqual(18, var_w.data[0])

    def test_extend_1D_vertical_to_2D(self):
        vertical_variable = xr.Variable('y', [5, 6, 7])
        _3d_variable = xr.Variable(('z', 'y', 'x'),
                                   [
                                       [
                                           [1, 2, 3, 4],
                                           [2, 3, 4, 5],
                                           [3, 4, 5, 6],
                                       ],
                                       [
                                           [4, 5, 6, 7],
                                           [5, 6, 7, 8],
                                           [6, 7, 8, 9],
                                       ],
                                   ])
        extended = RD._extend_1D_vertical_to_2D(vertical_variable, _3d_variable)
        self.assertEqual((3, 4), extended.shape)
        self.assertEqual(('y', 'x'), extended.dims)
        expected = np.asarray([[5, 5, 5, 5],
                               [6, 6, 6, 6],
                               [7, 7, 7, 7], ])
        ftu.assert_array_equals_with_index_error_message(self, expected, extended.data)
