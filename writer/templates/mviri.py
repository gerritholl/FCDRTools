import numpy as np
from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil

SWATH_WIDTH = 4000
SRF_SIZE = 176
SOL_IRR_SIZE = 24

class MVIRI:
    @staticmethod
    def add_original_variables(dataset, height):
        TemplateUtil.add_geolocation_variables(dataset, SWATH_WIDTH, height)

        # time
        default_array = DefaultData.create_default_vector(height, np.int32)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int32)
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time in seconds since 1970-01-01 00:00:00"
        variable.attrs["units"] = "s"
        dataset["time"] = variable

        # time_delta
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int8)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int8)
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time delta"
        variable.attrs["units"] = "s"
        variable.attrs["scale_factor"] = 0.025
        dataset["time_delta"] = variable

        # satellite_azimuth_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "sensor_azimuth_angle"
        variable.attrs["units"] = "degree"
        dataset["satellite_azimuth_angle"] = variable

        # satellite_zenith_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "sensor_zenith_angle"
        variable.attrs["units"] = "degree"
        dataset["satellite_zenith_angle"] = variable

        # solar_azimuth_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "solar_azimuth_angle"
        variable.attrs["units"] = "degree"
        dataset["solar_azimuth_angle"] = variable

        # solar_zenith_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "solar_zenith_angle"
        variable.attrs["units"] = "degree"
        dataset["solar_zenith_angle"] = variable

        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "Image counts"
        variable.attrs["units"] = "count"
        dataset["count"] = variable

    @staticmethod
    def get_swath_width():
        return SWATH_WIDTH

    @staticmethod
    def add_uncertainty_variables(dataset, height):
        # srf
        default_array = DefaultData.create_default_vector(SRF_SIZE, np.float32)
        variable = Variable(["srf_size"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Spectral Response Function"
        dataset["srf"] = variable

        # a0
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Calibration Coefficient at Launch"
        dataset["a0"] = variable

        # a1
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Time variation of a0"
        dataset["a1"] = variable

        # sol_irr
        default_array = DefaultData.create_default_vector(SOL_IRR_SIZE, np.float32)
        variable = Variable(["sol_irr_size"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Solar Irradiance"
        dataset["sol_irr"] = variable
