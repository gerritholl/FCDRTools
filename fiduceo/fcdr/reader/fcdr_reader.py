import xarray as xr


class FCDRReader:

    @classmethod
    def read(cls, file_str, drop_variables_str=None, decode_cf=True, decode_times=True,
             engine_str=None):
        """
        Read a dataset from a netCDF 3/4 or HDF file.

        :param file: The netCDF file path.
        :param drop_variables: List of variables to be dropped.
        :param decode_cf: Whether to decode CF attributes and coordinate variables.
        :param decode_times: Whether to decode time information (convert time coordinates to ``datetime`` objects).
        :param engine: Optional netCDF engine name.
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
            values = evaluate(expression_, dic)
            dims = dic.copy().popitem()[1]._dims
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
    def _get_biggest_dimensions(cls, dic, expression):
        bigest_shape_product = 0
        dims = None
        skeys = cls._get_keys_sortest_longest_first(dic)
        for key in skeys:
            if key in expression:
                expression = expression.replace(key, '$$')
                variable = dic[key]
                shape_product = cls._get_shape_product(variable)
                if shape_product > bigest_shape_product:
                    bigest_shape_product = shape_product
                    dims = variable.dims
        return dims


    @classmethod
    def _get_keys_sortest_longest_first(cls, dic):
        dic_keys = dic.viewkeys()
        return sorted(dic_keys, key=len, reverse=True)


    @classmethod
    def _find_used_one_d_variables_to_expand(cls, dic, dims, expression):
        one_dimensional_variables = {}
        dim_len = len(dims)
        if dim_len == 1:
            return one_dimensional_variables
        dim_of_interest = dims[dim_len - 2]
        sorted_keys = cls._get_keys_sortest_longest_first(dic)
        for key in sorted_keys:
            if key in expression:
                expression.replace(key, '$$')
                variable = dic[key]
                if len(variable.shape) == 1 and variable.dims[0] == dim_of_interest:
                    one_dimensional_variables.update({key:variable})
        return one_dimensional_variables
