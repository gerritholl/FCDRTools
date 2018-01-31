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
        cls._prepare_virtual_variables(ds)
        return ds

    @classmethod
    def _create_dictionary_of_non_virtuals(cls, ds):
        dic = {}
        for varName in ds._variables:
            if "virtual" not in ds._variables[varName].attrs:
                dic.update({varName: ds._variables[varName]})
        return dic

    @classmethod
    def _get_virtual_lazy_load(cls, ds, var_name):
        from numexpr import evaluate
        from xarray import Variable

        def _virtual_lazy_load(self):
            dic = cls._create_dictionary_of_non_virtuals(ds)
            expression_ = self.attrs["expression"]
            biggest_variable = cls._get_biggest_variable(dic, expression_)
            dims = biggest_variable.dims
            to_extend = cls._find_used_one_dimensional_variables_to_extend(dic, dims, expression_)
            for name in to_extend:
                dic[name] = cls._extend_1D_vertical_to_2D(dic[name], biggest_variable)
            values = evaluate(expression_, dic)
            tmp_var = Variable(dims, values)
            tmp_var.attrs = self.attrs
            ds._variables[var_name] = tmp_var
            return ds._variables[var_name]

        return _virtual_lazy_load

    @classmethod
    def _prepare_virtual_variables(cls, ds):
        for varName in ds._variables:
            if "virtual" in ds._variables[varName].attrs:
                import types
                ds._variables[varName].load = types.MethodType(cls._get_virtual_lazy_load(ds, varName), ds._variables[varName])

        return ds

    @classmethod
    def _get_shape_product(cls, variable):
        shape = variable.shape
        max = len(shape)
        prod = shape[0]
        for i in range(1, max):
            prod = shape[i] * prod
        return prod

    @classmethod
    def _get_biggest_variable(cls, dic, expression):
        bigest_shape_product = 0
        bigest_var = None
        skeys = cls._get_keys_sorted__longest_first(dic)
        for key in skeys:
            if key in expression:
                expression = expression.replace(key, '$$')
                variable = dic[key]
                shape_product = cls._get_shape_product(variable)
                if shape_product > bigest_shape_product:
                    bigest_shape_product = shape_product
                    bigest_var = variable
        return bigest_var

    @classmethod
    def _get_keys_sorted__longest_first(cls, dic):
        dic_keys = dic.viewkeys()
        return sorted(dic_keys, key=len, reverse=True)

    @classmethod
    def _find_used_one_dimensional_variables_to_extend(cls, dic, biggest_dims, expression):
        one_dimensional_variables = list()
        dim_len = len(biggest_dims)
        if dim_len == 1:
            return one_dimensional_variables
        dim_of_interest = biggest_dims[dim_len - 2]
        sorted_keys = cls._get_keys_sorted__longest_first(dic)
        for key in sorted_keys:
            if key in expression:
                expression.replace(key, '')
                variable = dic[key]
                if len(variable.shape) == 1 and variable.dims[0] == dim_of_interest:
                    one_dimensional_variables.append(key)
        return one_dimensional_variables

    @classmethod
    def _extend_1D_vertical_to_2D(cls, vertical_variable, reference_var):
        shape = reference_var.shape[-2:]
        var_reshaped = np.resize(vertical_variable, shape[::-1])
        var_reshaped = np.moveaxis(var_reshaped, 0, 1)
        return xr.Variable(reference_var.dims[-2:], var_reshaped)
