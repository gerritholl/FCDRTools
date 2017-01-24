import numpy as np
from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil

SWATH_WIDTH = 409

class AVHRR:
    @staticmethod
    def add_original_variables(dataset, height):
        TemplateUtil.add_geolocation_variables(dataset, SWATH_WIDTH, height)

        # Time
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time in seconds since 1970-01-01 00:00:00"
        variable.attrs["units"] = "s"
        dataset["Time"] = variable

        #scanline
        default_array = DefaultData.create_default_vector(height, np.int16)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "scanline"
        variable.attrs["long_name"] = "Level 1b line number"
        variable.attrs["valid_min"] = 0
        dataset["scanline"] = variable

        # satellite_azimuth_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "sensor_azimuth_angle"
        variable.attrs["units"] = "degree"
        dataset["satellite_azimuth_angle"] = variable

        # satellite_zenith_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "sensor_zenith_angle"
        variable.attrs["add_offset"] = 0.0
        variable.attrs["scale_factor"] = 0.01
        variable.attrs["units"] = "degree"
        variable.attrs["valid_max"] = 9000
        variable.attrs["valid_min"] = 0
        dataset["satellite_zenith_angle"] = variable

        # solar_azimuth_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "solar_azimuth_angle"
        variable.attrs["units"] = "degree"
        dataset["solar_azimuth_angle"] = variable

        # solar_zenith_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "solar_zenith_angle"
        variable.attrs["add_offset"] = 0.0
        variable.attrs["scale_factor"] = 0.01
        variable.attrs["units"] = "degree"
        variable.attrs["valid_max"] = 18000
        variable.attrs["valid_min"] = 0
        dataset["solar_zenith_angle"] = variable

        # Ch1_Bt
        variable = AVHRR.create_channel_refl_variable(height, "Channel 1 Reflectance")
        dataset["Ch1_Bt"] = variable

        # Ch2_Bt
        variable = AVHRR.create_channel_refl_variable(height, "Channel 2 Reflectance")
        dataset["Ch2_Bt"] = variable

        # Ch3a_Bt
        variable = AVHRR.create_channel_refl_variable(height, "Channel 3a Reflectance")
        dataset["Ch3a_Bt"] = variable

        # Ch3b_Bt
        variable = AVHRR.create_channel_bt_variable(height, "Channel 3b Brightness Temperature")
        dataset["Ch3b_Bt"] = variable

        # Ch4_Bt
        variable = AVHRR.create_channel_bt_variable(height, "Channel 4 Brightness Temperature")
        dataset["Ch4_Bt"] = variable

        # Ch5_Bt
        variable = AVHRR.create_channel_bt_variable(height, "Channel 5 Brightness Temperature")
        dataset["Ch5_Bt"] = variable

        default_array = DefaultData.create_default_vector(height, np.int16)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "Temperature of the internal calibration target"
        AVHRR.add_temperature_attributes(variable)
        dataset["T_ICT"] = variable

    @staticmethod
    def get_swath_width():
        return SWATH_WIDTH

    @staticmethod
    def add_temperature_attributes(variable):
        variable.attrs["add_offset"] = 273.15
        variable.attrs["scale_factor"] = 0.01
        variable.attrs["units"] = "kelvin"
        variable.attrs["valid_max"] = 10000
        variable.attrs["valid_min"] = -20000

    @staticmethod
    def create_channel_refl_variable(height, long_name):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "toa_reflectance"
        variable.attrs["long_name"] = long_name
        variable.attrs["add_offset"] = 0.0
        variable.attrs["scale_factor"] = 1e-4
        variable.attrs["units"] = "percent"
        variable.attrs["valid_max"] = 15000
        variable.attrs["valid_min"] = 0
        return variable

    @staticmethod
    def create_channel_bt_variable(height, long_name):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "toa_brightness_temperature"
        variable.attrs["long_name"] = long_name
        AVHRR.add_temperature_attributes(variable)
        return variable
