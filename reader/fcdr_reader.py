

import xarray as xr


class FCDRReader:

    def read(file: str, drop_variables: str = None, decode_cf: bool = True, decode_times: bool = True,
             engine: str = None) -> xr.Dataset:
        """
        Read a dataset from a netCDF 3/4 or HDF file.

        :param file: The netCDF file path.
        :param drop_variables: List of variables to be dropped.
        :param decode_cf: Whether to decode CF attributes and coordinate variables.
        :param decode_times: Whether to decode time information (convert time coordinates to ``datetime`` objects).
        :param engine: Optional netCDF engine name.
        """

        return xr.open_dataset(file, drop_variables=drop_variables,
                               decode_cf=decode_cf, decode_times=decode_times, engine=engine)
