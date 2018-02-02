import xarray as xr
import numpy as np


class FCDRReader:

    @classmethod
    def read(cls, file_str, drop_variables_str=None, decode_cf=True, decode_times=True,
             engine_str=None):
        """Read a dataset from a netCDF 3/4 or HDF file.

        Parameters
        ----------
        file_str: str
            The netCDF file path.
        drop_variables_str: str or iterable, optional
            List of variables to be dropped.
        decode_cf: bool, optional
            Whether to decode CF attributes and coordinate variables.
        decode_times: bool, optional
            Whether to decode time information (convert time coordinates to ``datetime`` objects).
        engine_str: str, optional
            Optional netCDF engine name.

        Return
        ------
        xarray.Dataset
        """
        ds = xr.open_dataset(file_str, drop_variables=drop_variables_str, decode_cf=decode_cf, decode_times=decode_times, engine=engine_str, chunks=1000000)
        return ds

    @classmethod
    def load_virtual_variable(cls, ds, var_name):

        import numexpr as ne
        import exceptions as ex

        v_var = ds.variables[var_name]
        if v_var is not None and "virtual" in v_var.attrs:
            if get_shape_product(v_var) <= 1:
                dic = create_dictionary_of_non_virtuals(ds)
                expression_ = v_var.attrs["expression"]
                biggest_variable = get_biggest_variable(dic, expression_)
                dims = biggest_variable.dims
                to_extend = find_used_one_dimensional_variables_to_extend(dic, dims, expression_)
                for name in to_extend:
                    dic[name] = extend_1d_vertical_to_2d(dic[name], biggest_variable)
                values = ne.evaluate(expression_, dic)
                tmp_var = xr.Variable(dims, values)
                tmp_var.attrs = v_var.attrs
                ds._variables[var_name] = tmp_var
        else:
            raise ex.ValueError('no such virtual variable: "' + var_name + '"')


def get_biggest_variable(dic, expression):
    bigest_shape_product = 0
    bigest_var = None
    skeys = get_keys_sorted__longest_first(dic)
    for key in skeys:
        if key in expression:
            expression = expression.replace(str(key), '')
            variable = dic[key]
            shape_product = get_shape_product(variable)
            if shape_product > bigest_shape_product:
                bigest_shape_product = shape_product
                bigest_var = variable
    return bigest_var


def get_shape_product(variable):
    shape = variable.shape
    shape_len = len(shape)
    prod = shape[0]
    for i in range(1, shape_len):
        prod = shape[i] * prod
    return prod


def find_used_one_dimensional_variables_to_extend(dic, biggest_dims, expression):
    one_dimensional_variables = list()
    dim_len = len(biggest_dims)
    if dim_len == 1:
        return one_dimensional_variables
    dim_of_interest = biggest_dims[dim_len - 2]
    sorted_keys = get_keys_sorted__longest_first(dic)
    for key in sorted_keys:
        if key in expression:
            expression.replace(str(key), '')
            variable = dic[key]
            if len(variable.shape) == 1 and variable.dims[0] == dim_of_interest:
                one_dimensional_variables.append(key)
    return one_dimensional_variables


def get_keys_sorted__longest_first(dic):
    dic_keys = dic.viewkeys()
    return sorted(dic_keys, key=len, reverse=True)


def extend_1d_vertical_to_2d(vertical_variable, reference_var):
    shape = reference_var.shape[-2:]
    var_reshaped = np.resize(vertical_variable, shape[::-1])
    var_reshaped = np.moveaxis(var_reshaped, 0, 1)
    return xr.Variable(reference_var.dims[-2:], var_reshaped)


def create_dictionary_of_non_virtuals(ds):
    dic = {}
    for varName in ds.variables:
        if "virtual" not in ds.variables[varName].attrs:
            dic.update({varName: ds.variables[varName]})
    return dic
