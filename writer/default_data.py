import numpy as np
import xarray as xr


class DefaultData:
    @staticmethod
    def create_default_array(width, height, dtype, fill_value=None):
        empty_array = np.empty([width, height], dtype)
        if fill_value is not None:
            empty_array.fill(fill_value)
        else:
            empty_array.fill(9.96921E36)
        default_array = xr.DataArray(empty_array, dims={'x': width, 'y': height})
        return default_array