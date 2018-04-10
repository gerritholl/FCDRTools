import numpy as np
from xarray import Variable, Coordinate

from fiduceo.fcdr.writer.default_data import DefaultData


class TemplateUtil:
    @staticmethod
    def add_geolocation_variables(dataset, width, height, chunksizes=None):
        default_array = DefaultData.create_default_array(width, height, np.float32, fill_value=np.NaN)

        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "latitude"
        TemplateUtil.add_units(variable, "degrees_north")
        TemplateUtil.add_encoding(variable, np.int16, -32768, scale_factor=0.0027466658, chunksizes=chunksizes)
        dataset["latitude"] = variable

        default_array = DefaultData.create_default_array(width, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "longitude"
        TemplateUtil.add_units(variable, "degrees_east")
        TemplateUtil.add_encoding(variable, np.int16, -32768, scale_factor=0.0054933317, chunksizes=chunksizes)
        dataset["longitude"] = variable

    @staticmethod
    def add_quality_flags(dataset, width, height, chunksizes=None, masks_append=None, meanings_append=None):
        default_array = DefaultData.create_default_array(width, height, np.uint8, fill_value=0)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "status_flag"

        masks = "1, 2, 4, 8, 16, 32, 64, 128"
        if masks_append is not None:
            masks = masks + masks_append
        variable.attrs["flag_masks"] = masks

        meanings = "invalid use_with_caution invalid_input invalid_geoloc invalid_time sensor_error padded_data incomplete_channel_data"
        if meanings_append is not None:
            meanings = meanings + meanings_append
        variable.attrs["flag_meanings"] = meanings

        if chunksizes is not None:
            TemplateUtil.add_chunking(variable, chunksizes)
        dataset["quality_pixel_bitmask"] = variable

    @staticmethod
    def create_scalar_float_variable(long_name=None, standard_name=None, units=None, fill_value=np.NaN):
        default_array = fill_value

        variable = Variable([], default_array)
        TemplateUtil.add_fill_value(variable, fill_value)

        if long_name is not None:
            variable.attrs["long_name"] = long_name

        if standard_name is not None:
            variable.attrs["standard_name"] = standard_name

        if units is not None:
            TemplateUtil.add_units(variable, units)
        return variable

    @staticmethod
    def create_float_variable(width, height, standard_name=None, long_name=None, dim_names=None, fill_value=None):
        if fill_value is None:
            default_array = DefaultData.create_default_array(width, height, np.float32)
        else:
            default_array = DefaultData.create_default_array(width, height, np.float32, fill_value=fill_value)

        if dim_names is None:
            variable = Variable(["y", "x"], default_array)
        else:
            variable = Variable(dim_names, default_array)

        if fill_value is None:
            variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        else:
            variable.attrs["_FillValue"] = fill_value

        if standard_name is not None:
            variable.attrs["standard_name"] = standard_name

        if long_name is not None:
            variable.attrs["long_name"] = long_name

        return variable

    @staticmethod
    def set_unsigned(variable):
        variable.attrs["_Unsigned"] = "true"

    @staticmethod
    def add_fill_value(variable, fill_value):
        variable.attrs["_FillValue"] = fill_value

    @staticmethod
    def add_units(variable, units):
        variable.attrs["units"] = units

    @staticmethod
    def add_scale_factor(variable, scale_factor):
        variable.attrs["scale_factor"] = scale_factor

    @staticmethod
    def add_offset(variable, offset):
        variable.attrs["add_offset"] = offset

    @staticmethod
    def add_encoding(variable, data_type, fill_value, scale_factor=1.0, offset=0.0, chunksizes=None):
        if chunksizes is None:
            variable.encoding = dict([('dtype', data_type), ('_FillValue', fill_value), ('scale_factor', scale_factor), ('add_offset', offset)])
        else:
            variable.encoding = dict([('dtype', data_type), ('_FillValue', fill_value), ('scale_factor', scale_factor), ('add_offset', offset), ('chunksizes', chunksizes)])

    @staticmethod
    def add_chunking(variable, chunksizes):
        variable.encoding = dict([('chunksizes', chunksizes)])

    @staticmethod
    def add_coordinates(ds):
        x_dim = ds.dims["x"]
        ds["x"] = Coordinate("x", np.arange(x_dim, dtype=np.uint16))

        y_dim = ds.dims["y"]
        ds["y"] = Coordinate("y", np.arange(y_dim, dtype=np.uint16))

        if "channel" in ds.dims:
            channels_dim = ds.dims["channel"]
            ds["channel"] = Coordinate("channel", np.arange(channels_dim, dtype=np.uint16))

    @staticmethod
    def add_correlation_matrices(dataset, num_channels):
        default_array = np.diag(np.ones(num_channels, dtype=np.float32))
        variable = Variable(["channel", "channel"], default_array)
        variable.attrs["long_name"] = "Channel_correlation_matrix_independent_effects"
        variable.attrs["units"] = "1"
        variable.attrs["description"] = "Channel error correlation matrix for independent effects"
        dataset['channel_correlation_matrix_independent'] = variable

        default_array = np.diag(np.ones(num_channels, dtype=np.float32))
        variable = Variable(["channel", "channel"], default_array)
        variable.attrs["long_name"] = "Channel_correlation_matrix_structured_effects"
        variable.attrs["units"] = "1"
        variable.attrs["description"] = "Channel error correlation matrix for structured effects"
        dataset['channel_correlation_matrix_structured'] = variable
