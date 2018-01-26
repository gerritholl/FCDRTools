import unittest as ut
import xarray as xr
import numpy as np
from fiduceo.fcdr.reader.fcdr_reader import FCDRReader


class FCDR_Reader_Test(ut.TestCase):

    def test_adding_2_three_dimensional_variables(self):
        ds = xr.Dataset()
        a = np.asarray([[[1, 2, 3], [4, 5, 6]],
                        [[11, 12, 13], [14, 15, 16]],
                        [[21, 22, 23], [24, 25, 26]]])
        ds['a'] = xr.Variable(['z', 'y', 'x'], a)
        ds['b'] = xr.Variable(['z', 'y', 'x'], a.copy())
        v_var = xr.Variable(["virtual"], np.full([1], 1))
        v_var.attrs["virtual"] = "true"
        v_var.attrs["expression"] = "a + b"
        ds["v_var"] = v_var

        self.assertEqual(3, len(ds.data_vars))
        FCDRReader._prepare_virtual_variables(ds)
        self.assertEqual(3, len(ds.data_vars))
        v1 = ds['v_var'].values
        v2 = np.full([1], 1)
        self.assertEqual(type(v1), type(v2))
        self.assertEqual(len(v1), len(v2))

        ds['v_var'].load()

        self.assertTrue('v_var' in ds)

        v1 = ds['v_var'].values
        v2 = np.asarray([[[2, 4, 6], [8, 10, 12]],
                         [[22, 24, 26], [28, 30, 32]],
                         [[42, 44, 46], [48, 50, 52]]])

        self.assertEqual(len(v1), len(v2))
        self.assertEqual("<type 'numpy.ndarray'>", str(type(v2)))
        self.assertEqual(type(v1), type(v2))
        self.assert_array_equals_with_index_error_message(v1, v2)

    def assert_array_equals_with_index_error_message(self, v1, v2):
        v1f = v1.flatten()
        v2f = v2.flatten()
        for idx, a in enumerate(v1f):
            self.assertEqual(a, v2f[idx], "values at index " + str(idx) + " are not equal.")
            idx += 1
