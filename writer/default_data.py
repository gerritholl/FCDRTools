import numpy as np
import xarray as xr


class DefaultData:
    @staticmethod
    def create_default_array(width, height, dtype, fill_value=None):
        empty_array = np.empty([height, width], dtype)
        if fill_value is not None:
            empty_array.fill(fill_value)
        else:
            empty_array.fill(9.96921E36)
        default_array = xr.DataArray(empty_array, dims=['y', 'x'])
        return default_array

    @staticmethod
    def create_default_array_3d(width, height, num_channels, dtype, fill_value=None):
        empty_array = np.empty([num_channels, height, width], dtype)
        if fill_value is not None:
            empty_array.fill(fill_value)
        else:
            empty_array.fill(9.96921E36)
        default_array = xr.DataArray(empty_array, dims=['channel', 'y', 'x'])
        return default_array
