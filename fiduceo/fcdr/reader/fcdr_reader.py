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
        from numexpr import NumExpr, disassemble, evaluate
        import xarray as xr
        from xarray import Variable, Dataset, DataArray
        from dask.dataframe import DataFrame, Series
        import numpy as np

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
