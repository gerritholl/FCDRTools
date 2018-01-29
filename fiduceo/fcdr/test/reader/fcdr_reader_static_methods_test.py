import unittest as ut
import xarray as xr
import numpy as np
from fiduceo.fcdr.reader.fcdr_reader import FCDRReader as RD


class FCDR_Reader_Static_Methods_Test(ut.TestCase):

    def setUp(self):
        ds = xr.Dataset()
        ds['asd'] = xr.Variable(['z', 'y', 'x'], np.full([3, 4, 5], 13))
        ds['lsmf'] = xr.Variable(['y', 'x'], np.full([4, 5], 14))
        ds['frasdsn'] = xr.Variable(['y', 'x'], np.full([4, 5], 15))
        ds['ka'] = xr.Variable(['x'], np.full([5], 16))
        ds['Y'] = xr.Variable(['y'], np.full([4], 17))
        self.ds = ds
        self.dic = RD._create_dictionary_of_non_virtuals(ds)


    def test_GetBiggestDimension(self):
        expression = '(asd)*[lsmf + frasdsn] - ka'
        dimensions = RD._get_biggest_dimensions(self.dic, expression)
        self.assertEqual(("z", "y", "x"), dimensions)

        expression = '[lsmf + frasdsn] - ka'
        dimensions = RD._get_biggest_dimensions(self.dic, expression)
        self.assertEqual(("y", "x"), dimensions)

        expression = 'lsmf +  ka'
        dimensions = RD._get_biggest_dimensions(self.dic, expression)
        self.assertEqual(("y", "x"), dimensions)

        expression = 'frasdsn - ka'
        dimensions = RD._get_biggest_dimensions(self.dic, expression)
        self.assertEqual(("y", "x"), dimensions)

        expression = 'ka * 4'
        dimensions = RD._get_biggest_dimensions(self.dic, expression)
        self.assertEqual(("x",), dimensions)


    def test_GetKeysSortestLongestFirst(self):
        dictonary = {"a": 1, "a2": 2, "b": 3, "bb3": 4, "c2": 5}
        sortedKeys = RD._get_keys_sortest_longest_first(dictonary)
        self.assertEqual(["bb3", "a2", "c2", "a", "b"], sortedKeys)


    def test_ExpandOneDimensionalVariables(self):
        pass

    def test_FindUsedOneDimensionalVerticalVarisable(self):
        expression = 'asd * ka'
        dims = RD._get_biggest_dimensions(self.dic, expression)
        one_d_vars = RD._find_used_one_d_variables_to_expand(self.dic, dims, expression)
        self.assertEquals({}, one_d_vars)

        expression = 'asd * Y + ka'
        one_d_vars = RD._find_used_one_d_variables_to_expand(self.dic, dims, expression)
        self.assertEquals(1, len(one_d_vars))
        self.assertTrue('Y' in one_d_vars)
        variable = one_d_vars['Y']
        self.assertEqual(('y',), variable.dims)
