import numpy as np
from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil

FULL_DIMENSION = 5000
IR_DIMENSION = 2500
SRF_SIZE = 176
SOL_IRR_SIZE = 24


class MVIRI:
    @staticmethod
    def add_original_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility
        # time
        default_array = DefaultData.create_default_vector(IR_DIMENSION, np.int32)
        variable = Variable(["y_ir"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int32)
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time in seconds since 1970-01-01 00:00:00"
        variable.attrs["units"] = "s"
        variable.attrs["_Unsigned"] = "true"
        dataset["time"] = variable

        # timedelta
        default_array = DefaultData.create_default_array(IR_DIMENSION, IR_DIMENSION, np.int16)
        variable = Variable(["y_ir", "x_ir"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Delta time at pixel acquisition against central pixel"
        variable.attrs["units"] = "s"
        variable.attrs["scale_factor"] = 0.001831083
        dataset["timedelta"] = variable

        dataset["satellite_azimuth_angle"] = MVIRI._create_angle_variable("sensor_azimuth_angle")
        dataset["satellite_zenith_angle"] = MVIRI._create_angle_variable("sensor_zenith_angle")
        dataset["solar_azimuth_angle"] = MVIRI._create_angle_variable("solar_azimuth_angle")
        dataset["solar_zenith_angle"] = MVIRI._create_angle_variable("solar_zenith_angle")

        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "Image counts"
        variable.attrs["units"] = "count"
        dataset["count"] = variable

    @staticmethod
    def get_swath_width():
        return FULL_DIMENSION

    @staticmethod
    def add_uncertainty_variables(dataset, height):
        # srf
        default_array = DefaultData.create_default_vector(SRF_SIZE, np.float32)
        variable = Variable(["srf_size"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Spectral Response Function"
        dataset["srf"] = variable

        # a0
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Calibration Coefficient at Launch"
        dataset["a0"] = variable

        # a1
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.float32)
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

        dataset["u_lat"] = MVIRI._create_angle_variable("Uncertainty in Latitude")
        dataset["u_lon"] = MVIRI._create_angle_variable("Uncertainty in Longitude")

        # u_time
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Uncertainty in Time"
        variable.attrs["units"] = "s"
        dataset["u_time"] = variable

        dataset["u_satellite_zenith_angle"] = MVIRI._create_angle_variable("Uncertainty in Satellite Zenith Angle")
        dataset["u_satellite_azimuth_angle"] = MVIRI._create_angle_variable("Uncertainty in Satellite Azimuth Angle")
        dataset["u_solar_zenith_angle"] = MVIRI._create_angle_variable("Uncertainty in Solar Zenith Angle")
        dataset["u_solar_azimuth_angle"] = MVIRI._create_angle_variable("Uncertainty in Solar Azimuth Angle")

        # u_tot_count
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Total Uncertainty in counts"
        variable.attrs["units"] = "count"
        dataset["u_tot_count"] = variable

        # u_srf
        default_array = DefaultData.create_default_vector(SRF_SIZE, np.float32)
        variable = Variable(["srf_size"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Uncertainty in SRF"
        dataset["u_srf"] = variable

        # u_a0
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Uncertainty in a0"
        dataset["u_a0"] = variable

        # u_a1
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Uncertainty in a1"
        dataset["u_a1"] = variable

        # u_sol_irr
        default_array = DefaultData.create_default_vector(SOL_IRR_SIZE, np.float32)
        variable = Variable(["sol_irr_size"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "Uncertainty in Solar Irradiance"
        dataset["u_sol_irr"] = variable

    @staticmethod
    def _create_angle_variable(standard_name):
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = standard_name
        variable.attrs["units"] = "degree"
        return variable
