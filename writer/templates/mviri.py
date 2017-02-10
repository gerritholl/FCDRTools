import numpy as np
from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil as tu

FULL_DIMENSION = 5000
IR_DIMENSION = 2500
SRF_SIZE = 1000
SOL_IRR_SIZE = 24


class MVIRI:
    @staticmethod
    def add_original_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05
        # time
        default_array = DefaultData.create_default_vector(IR_DIMENSION, np.int32)
        variable = Variable(["y_ir"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int32))
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time in seconds since 1970-01-01 00:00:00"
        variable.attrs["units"] = "s"
        tu.set_unsigned(variable)
        dataset["time"] = variable

        # timedelta
        default_array = DefaultData.create_default_array(IR_DIMENSION, IR_DIMENSION, np.int16)
        variable = Variable(["y_ir", "x_ir"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Delta time at pixel acquisition against central pixel"
        variable.attrs["units"] = "s"
        variable.attrs["scale_factor"] = 0.001831083
        dataset["timedelta"] = variable

        dataset["satellite_azimuth_angle"] = MVIRI._create_angle_variable_int(0.005493164,
                                                                              standard_name="sensor_azimuth_angle")
        tu.set_unsigned(dataset["satellite_azimuth_angle"])
        dataset["satellite_zenith_angle"] = MVIRI._create_angle_variable_int(0.005493248,
                                                                             standard_name="sensor_zenith_angle")
        dataset["solar_azimuth_angle"] = MVIRI._create_angle_variable_int(0.005493164,
                                                                          standard_name="solar_azimuth_angle")
        tu.set_unsigned(dataset["solar_azimuth_angle"])
        dataset["solar_zenith_angle"] = MVIRI._create_angle_variable_int(0.005493248,
                                                                         standard_name="solar_zenith_angle")

        # count
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int8)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int8))
        variable.attrs["long_name"] = "Image counts"
        variable.attrs["units"] = "count"
        tu.set_unsigned(variable)
        dataset["count"] = variable

        # reflectance
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["standard_name"] = "toa_reflectance"
        variable.attrs["units"] = "percent"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 1.52588E-05
        dataset["reflectance"] = variable

        # srf
        default_array = DefaultData.create_default_vector(SRF_SIZE, np.float32)
        variable = Variable(["srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Spectral Response Function"
        dataset["srf"] = variable

    @staticmethod
    def get_swath_width():
        return FULL_DIMENSION

    @staticmethod
    def add_uncertainty_variables(dataset, height):
        # sol_irr
        default_array = DefaultData.create_default_vector(SRF_SIZE, np.float32)
        variable = Variable(["srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "solar_irradiance_per_unit_wavelength"
        variable.attrs["units"] = "W*m-2*nm-1"
        dataset["sol_irr"] = variable

        # sol_eff_irr
        default_array = DefaultData.get_default_fill_value(np.float32)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Solar effective Irradiance"
        variable.attrs["units"] = "W*m-2"
        dataset["sol_eff_irr"] = variable

        dataset["u_latitude"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Latitude",
                                                                 unsigned=True)
        dataset["u_longitude"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Longitude",
                                                                  unsigned=True)

        # u_time
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.float32)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "Uncertainty in Time"
        variable.attrs["units"] = "s"
        tu.set_unsigned(variable)
        dataset["u_time"] = variable

        dataset["u_satellite_zenith_angle"] = MVIRI._create_angle_variable_int(7.62939E-05,
                                                                               long_name="Uncertainty in Satellite Zenith Angle",
                                                                               unsigned=True)
        dataset["u_satellite_azimuth_angle"] = MVIRI._create_angle_variable_int(7.62939E-05,
                                                                                long_name="Uncertainty in Satellite Azimuth Angle",
                                                                                unsigned=True)
        dataset["u_solar_zenith_angle"] = MVIRI._create_angle_variable_int(7.62939E-05,
                                                                           long_name="Uncertainty in Solar Zenith Angle",
                                                                           unsigned=True)
        dataset["u_solar_azimuth_angle"] = MVIRI._create_angle_variable_int(7.62939E-05,
                                                                            long_name="Uncertainty in Solar Azimuth Angle",
                                                                            unsigned=True)

        # u_tot_count
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Total Uncertainty in counts"
        variable.attrs["units"] = "count"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 7.62939E-05
        dataset["u_tot_count"] = variable

        # u_srf
        default_array = DefaultData.create_default_array(SRF_SIZE, SRF_SIZE, np.int16)
        variable = Variable(["srf_size", "srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in SRF"
        variable.attrs["scale_factor"] = 1.52588E-05
        tu.set_unsigned(variable)
        dataset["u_srf"] = variable

        # a0
        default_array = DefaultData.get_default_fill_value(np.float32)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Calibration Coefficient at Launch"
        variable.attrs["units"] = "Wm^-2sr^-1/count"
        dataset["a0"] = variable

        # a1
        default_array = DefaultData.get_default_fill_value(np.float32)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Time variation of a0"
        variable.attrs["units"] = "Wm^-2sr^-1/count day^-1 10^5"
        dataset["a1"] = variable

        # dSE
        default_array = DefaultData.get_default_fill_value(np.int8)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int8))
        variable.attrs["long_name"] = "Sun-Earth distance"
        variable.attrs["units"] = "au"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 0.00390625
        dataset["dSE"] = variable

        # K_space
        default_array = DefaultData.get_default_fill_value(np.int8)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int8))
        variable.attrs["long_name"] = "Space count"
        variable.attrs["units"] = "count"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 0.00390625
        dataset["K_space"] = variable

        # u_a0
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in a0"
        variable.attrs["units"] = "Wm^-2sr^-1/count"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 1.52588E-05
        dataset["u_a0"] = variable

        # u_a1
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in a1"
        variable.attrs["units"] = "Wm^-2sr^-1/count day^-1 10^5"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 0.000762939
        dataset["u_a1"] = variable

        # u_sol_eff_irr
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in Solar effective Irradiance"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 0.001525879
        dataset["u_sol_eff_irr"] = variable

        # u_shot-noise
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty due to shot noise"
        variable.attrs["units"] = "count"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 7.62939E-05
        dataset["u_shot-noise"] = variable

        # u_dSE
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in Sun-Earth distance"
        variable.attrs["units"] = "au"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 7.62939E-05
        dataset["u_dSE"] = variable

        # u_e-noise
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty due to Electronics noise"
        variable.attrs["units"] = "count"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 7.62939E-05
        dataset["u_e-noise"] = variable

        # u_digitization
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty due to digitization"
        variable.attrs["units"] = "count"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 7.62939E-05
        dataset["u_digitization"] = variable

        # u_space
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty of space count"
        variable.attrs["units"] = "count"
        tu.set_unsigned(variable)
        variable.attrs["scale_factor"] = 7.62939E-05
        dataset["u_space"] = variable

    @staticmethod
    def _create_angle_variable_int(scale_factor, standard_name=None, long_name=None, unsigned=False):
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        if standard_name is not None:
            variable.attrs["standard_name"] = standard_name

        if long_name is not None:
            variable.attrs["long_name"] = long_name

        if unsigned is True:
            tu.set_unsigned(variable)

        variable.attrs["units"] = "degree"
        variable.attrs["scale_factor"] = scale_factor
        return variable
