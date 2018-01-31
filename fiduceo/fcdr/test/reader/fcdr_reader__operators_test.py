import unittest as ut
import xarray as xr
import numpy as np
import fiduceo.fcdr.test.test_utils as tu
from fiduceo.fcdr.reader.fcdr_reader import FCDRReader as Reader
import math

t, f = True, False
pi = math.pi


class FCDR_Reader_Operators_Test(ut.TestCase):

    def test_and(self):
        expression = 'a & b'  # the meaning is "a and b"
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_one_dim_bool_variable()
        ds['b'] = get_two_dim_bool_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray([[t, f, f, f], [f, f, f, f], [t, f, f, f]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_or(self):
        expression = 'a | b'  # the meaning is "a or b"
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_one_dim_bool_variable()
        ds['b'] = get_two_dim_bool_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray([[t, f, t, t], [t, f, t, t], [t, f, t, f]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_not(self):
        expression = '~ a'  # the meaning is "not a"
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_one_dim_bool_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['x'], np.asarray([f, t, f, t]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_and_with_vertical_1D(self):
        expression = 'a & b'  # the meaning is "a and b"
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_vertical_one_dim_bool_variable()
        ds['b'] = get_two_dim_bool_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray([[f, f, f, f], [f, f, f, t], [f, f, f, f]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_or_with_vertical_1D(self):
        expression = 'a | b'  # the meaning is "a or b"
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_vertical_one_dim_bool_variable()
        ds['b'] = get_two_dim_bool_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray([[t, f, f, t], [t, t, t, t], [t, f, f, f]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_not_with_vertical_1D(self):
        expression = '~ a'  # the meaning is "not a"
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_vertical_one_dim_bool_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y'], np.asarray([t, f, t]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_less_than(self):
        expression = 'a < 5'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_three_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['z', 'y', 'x'], np.asarray(
            [[[t, t, t, t], [t, t, t, f], [t, t, f, f]],
             [[t, f, f, f], [f, f, f, f], [f, f, f, f]]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_less_than_or_equal(self):
        expression = 'a <= 5'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_three_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['z', 'y', 'x'], np.asarray(
            [[[t, t, t, t], [t, t, t, t], [t, t, t, f]],
             [[t, t, f, f], [t, f, f, f], [f, f, f, f]]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_equal(self):
        expression = 'a == 5'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_three_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['z', 'y', 'x'], np.asarray(
            [[[f, f, f, f], [f, f, f, t], [f, f, t, f]],
             [[f, t, f, f], [t, f, f, f], [f, f, f, f]]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_not_equal(self):
        expression = 'a != 5'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_three_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['z', 'y', 'x'], np.asarray(
            [[[t, t, t, t], [t, t, t, f], [t, t, f, t]],
             [[t, f, t, t], [f, t, t, t], [t, t, t, t]]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_greater_than_or_equal(self):
        expression = 'a >= 5'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_three_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['z', 'y', 'x'], np.asarray(
            [[[f, f, f, f], [f, f, f, t], [f, f, t, t]],
             [[f, t, t, t], [t, t, t, t], [t, t, t, t]]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_greater_than(self):
        expression = 'a > 5'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_three_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['z', 'y', 'x'], np.asarray(
            [[[f, f, f, f], [f, f, f, f], [f, f, f, t]],
             [[f, f, t, t], [f, t, t, t], [t, t, t, t]]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_unary_negation(self):
        expression = '- a'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_two_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray(
            [[-10, -20, -30, -40], [-13, -23, -33, -43], [-16, -26, -36, -46]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_addition(self):
        expression = 'a + b'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_two_dim_int_variable()
        ds['b'] = get_vertical_one_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray(
            [[210, 220, 230, 240], [313, 323, 333, 343], [416, 426, 436, 446]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_subtraction(self):
        expression = 'b - a'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_two_dim_int_variable()
        ds['b'] = get_vertical_one_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray(
            [[190, 180, 170, 160], [287, 277, 267, 257], [384, 374, 364, 354]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_multiplication(self):
        expression = 'a * b'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_two_dim_int_variable()
        ds['b'] = get_vertical_one_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray(
            [[2000, 4000, 6000, 8000], [3900, 6900, 9900, 12900], [6400, 10400, 14400, 18400]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_division(self):
        expression = 'b / a'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_two_dim_int_variable()
        ds['b'] = get_vertical_one_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray(
            [[20, 10, 6, 5], [23, 13, 9, 6], [25, 15, 11, 8]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_power(self):
        expression = 'a ** 2'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_two_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray(
            [[100, 400, 900, 1600], [169, 529, 1089, 1849], [256, 676, 1296, 2116]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_modulo(self):
        expression = 'a % 7'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_two_dim_int_variable()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['y', 'x'], np.asarray(
            [[3, 6, 2, 5], [6, 2, 5, 1], [2, 5, 1, 4]]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_sinus(self):
        expression = 'sin(a)'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_one_dim_radians()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['x'], np.asarray(
            [0,
             0.70710678,
             0.989821441881,
             0.70710678,
             -0.70710678,
             -0.989821441881]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_cosine(self):
        expression = 'cos(a)'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_one_dim_radians()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['x'], np.asarray(
            [1,
             0.70710678,
             0.142314838273,
             -0.70710678,
             -0.70710678,
             -0.142314838273]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_tangens(self):
        expression = 'tan(a)'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = get_one_dim_radians()
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected = xr.Variable(['x'], np.asarray(
            [0,
             1,
             6.95515277177,
             -1,
             1,
             6.95515277177]))
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_inverse_sinus(self):
        expression = 'arcsin(sin(a))'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = xr.Variable(('x'), [0, 0.5, 1])
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected =ds['a']
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_inverse_cosine(self):
        expression = 'arccos(cos(a))'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = xr.Variable(('x'), [1, 0.5, 0])
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected =ds['a']
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])

    def test_inverse_tangens(self):
        expression = 'arctan(tan(a))'
        ds = xr.Dataset()
        ds['v'] = create_virtual_variable(expression)
        ds['a'] = xr.Variable(('x'), [1, 0.5, 0])
        Reader._prepare_virtual_variables(ds)
        ds['v'].load()
        expected =ds['a']
        tu.assert_array_equals_with_index_error_message(self, expected, ds['v'])


def get_one_dim_radians():
    pi_4 = pi / 4
    pi_2 = pi / 2.2
    return xr.Variable(['x'], np.asarray([0,
                                          pi_4,
                                          pi_2,
                                          pi - pi_4,
                                          pi + pi_4,
                                          pi + pi_2]))


def get_three_dim_bool_variable():
    return xr.Variable(['z', 'y', 'x'], np.asarray(
        [[[t, f, f, t], [f, f, f, t], [t, f, f, f]],
         [[f, t, t, f], [t, t, t, f], [f, t, t, t]]]))


def get_two_dim_bool_variable():
    return xr.Variable(['y', 'x'], np.asarray(
        [[t, f, f, t], [f, f, f, t], [t, f, f, f]]))


def get_one_dim_bool_variable():
    return xr.Variable(['x'], np.asarray(
        [t, f, t, f]))


def get_vertical_one_dim_bool_variable():
    return xr.Variable(['y'], np.asarray(
        [f, t, f]))


def get_three_dim_int_variable():
    return xr.Variable(['z', 'y', 'x'], np.asarray(
        [[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
         [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]]))


def get_two_dim_int_variable():
    return xr.Variable(['y', 'x'], np.asarray(
        [[10, 20, 30, 40], [13, 23, 33, 43], [16, 26, 36, 46]]))


def get_one_dim_int_variable():
    return xr.Variable(['x'], np.asarray(
        [60, 70, 80, 90]))


def get_vertical_one_dim_int_variable():
    return xr.Variable(['y'], np.asarray(
        [200, 300, 400]))


def create_virtual_variable(expression):
    v_var = xr.Variable(["virtual"], np.full([1], 1))
    v_var.attrs["virtual"] = "true"
    v_var.attrs["expression"] = expression
    return v_var
