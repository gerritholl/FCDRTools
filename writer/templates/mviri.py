import numpy as np
from xarray import Variable

from writer.correlation import Correlation as corr
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
        default_array = DefaultData.create_default_array(IR_DIMENSION, IR_DIMENSION, np.int32)
        variable = Variable(["y_ir", "x_ir"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int32))
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time of pixel"
        tu.add_units(variable, "seconds since 1970-01-01 00:00:00")
        tu.set_unsigned(variable)
        dataset["time"] = variable

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

        # reflectance
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["standard_name"] = "toa_bidirectional_reflectance"
        tu.add_units(variable, "percent")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 1.52588E-05)
        dataset["reflectance"] = variable

        # srf
        default_array = DefaultData.create_default_vector(SRF_SIZE, np.float32)
        variable = Variable(["srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Spectral Response Function"
        dataset["spectral_response_function_vis"] = variable

    @staticmethod
    def get_swath_width():
        return FULL_DIMENSION

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05
        # u_random
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["long_name"] = "random uncertainty per pixel"
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        tu.set_unsigned(variable)
        tu.add_units(variable, "percent")
        tu.add_scale_factor(variable, 1.52588E-05)
        dataset["u_random"] = variable

        # u_non_random
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["long_name"] = "non-random uncertainty per pixel"
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        tu.set_unsigned(variable)
        tu.add_units(variable, "percent")
        tu.add_scale_factor(variable, 1.52588E-05)
        dataset["u_non_random"] = variable

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05

        # count_vis
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int8)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int8))
        variable.attrs["long_name"] = "Image counts"
        tu.add_units(variable, "count")
        tu.set_unsigned(variable)
        dataset["count_vis"] = variable

        # sol_irr
        default_array = DefaultData.create_default_vector(SRF_SIZE, np.float32)
        variable = Variable(["srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "solar_irradiance_per_unit_wavelength"
        tu.add_units(variable, "W*m-2*nm-1")
        dataset["sol_irr"] = variable

        # sol_eff_irr
        default_array = DefaultData.get_default_fill_value(np.float32)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Solar effective Irradiance"
        tu.add_units(variable, "W*m-2")
        dataset["sol_eff_irr"] = variable

        dataset["u_latitude"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Latitude",
                                                                 unsigned=True)
        MVIRI._add_geo_correlation_attributes(dataset["u_latitude"])

        dataset["u_longitude"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Longitude",
                                                                  unsigned=True)
        MVIRI._add_geo_correlation_attributes(dataset["u_longitude"])

        # u_time
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.float32)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "Uncertainty in Time"
        tu.add_units(variable, "s")
        tu.set_unsigned(variable)
        variable.attrs["pdf_shape"] = "rectangle"
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
        tu.add_units(variable, "count")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 7.62939E-05)
        variable.attrs[corr.SCAN_CORR_FORM] = corr.EIFFEL
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.PIXEL
        variable.attrs[corr.SCAN_CORR_SCALE] = [-2, 2]
        variable.attrs[corr.TIME_CORR_FORM] = corr.EIFFEL
        variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        variable.attrs[corr.TIME_CORR_SCALE] = [-2, 2]
        variable.attrs["pdf_shape"] = "digitised_gaussian"
        dataset["u_tot_count"] = variable

        # u_srf
        default_array = DefaultData.create_default_array(SRF_SIZE, SRF_SIZE, np.int16)
        variable = Variable(["srf_size", "srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in SRF"
        tu.add_scale_factor(variable, 1.52588E-05)
        tu.set_unsigned(variable)
        variable.attrs[corr.SCAN_CORR_FORM] = corr.RECT
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.PIXEL
        variable.attrs[corr.SCAN_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.TIME_CORR_FORM] = corr.RECT
        variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        variable.attrs[corr.TIME_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.IMG_CORR_FORM] = corr.RECT
        variable.attrs[corr.IMG_CORR_UNIT] = corr.DAYS
        variable.attrs[corr.IMG_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs["pdf_shape"] = "rectangle"
        dataset["u_srf"] = variable

        # a0
        default_array = DefaultData.get_default_fill_value(np.float32)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Calibration Coefficient at Launch"
        tu.add_units(variable, "Wm^-2sr^-1/count")
        dataset["a0"] = variable

        # a1
        default_array = DefaultData.get_default_fill_value(np.float32)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Time variation of a0"
        tu.add_units(variable, "Wm^-2sr^-1/count day^-1 10^5")
        dataset["a1"] = variable

        # dSE
        default_array = DefaultData.get_default_fill_value(np.int8)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int8))
        variable.attrs["long_name"] = "Sun-Earth distance"
        tu.add_units(variable, "au")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 0.00390625)
        dataset["dSE"] = variable

        # K_space
        default_array = DefaultData.get_default_fill_value(np.int8)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int8))
        variable.attrs["long_name"] = "Space count"
        tu.add_units(variable, "count")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 0.00390625)
        dataset["K_space"] = variable

        # u_a0
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in a0"
        tu.add_units(variable, "Wm^-2sr^-1/count")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 1.52588E-05)
        MVIRI._add_calibration_coeff_correlation_attributes(variable)
        dataset["u_a0"] = variable

        # u_a1
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in a1"
        tu.add_units(variable, "Wm^-2sr^-1/count day^-1 10^5")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 0.000762939)
        MVIRI._add_calibration_coeff_correlation_attributes(variable)
        dataset["u_a1"] = variable

        # u_sol_eff_irr
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in Solar effective Irradiance"
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 0.001525879)
        variable.attrs[corr.SCAN_CORR_FORM] = corr.RECT
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.PIXEL
        variable.attrs[corr.SCAN_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.TIME_CORR_FORM] = corr.RECT
        variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        variable.attrs[corr.TIME_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.IMG_CORR_FORM] = corr.RECT
        variable.attrs[corr.IMG_CORR_UNIT] = corr.DAYS
        variable.attrs[corr.IMG_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs["pdf_shape"] = "rectangle"
        dataset["u_sol_eff_irr"] = variable

        # u_shot-noise
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty due to shot noise"
        tu.add_units(variable, "count")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 7.62939E-05)
        dataset["u_shot-noise"] = variable

        # u_dSE
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty in Sun-Earth distance"
        tu.add_units(variable, "au")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 7.62939E-05)
        dataset["u_dSE"] = variable

        # u_e-noise
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty due to Electronics noise"
        tu.add_units(variable, "count")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 7.62939E-05)
        dataset["u_e-noise"] = variable

        # u_digitization
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty due to digitization"
        tu.add_units(variable, "count")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 7.62939E-05)
        dataset["u_digitization"] = variable

        # u_space
        default_array = DefaultData.get_default_fill_value(np.int16)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Uncertainty of space count"
        tu.add_units(variable, "count")
        tu.set_unsigned(variable)
        tu.add_scale_factor(variable, 7.62939E-05)
        variable.attrs[corr.TIME_CORR_FORM] = corr.RECT
        variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        variable.attrs[corr.TIME_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs["pdf_shape"] = "digitised_gaussian"
        dataset["u_space"] = variable

    @staticmethod
    def _add_geo_correlation_attributes(geo_variable):
        geo_variable.attrs[corr.SCAN_CORR_FORM] = corr.TRI
        geo_variable.attrs[corr.SCAN_CORR_UNIT] = corr.PIXEL
        geo_variable.attrs[corr.SCAN_CORR_SCALE] = [-250, 250]
        geo_variable.attrs[corr.TIME_CORR_FORM] = corr.TRI
        geo_variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        geo_variable.attrs[corr.TIME_CORR_SCALE] = [-250, 250]
        geo_variable.attrs[corr.IMG_CORR_FORM] = corr.TRI
        geo_variable.attrs[corr.IMG_CORR_UNIT] = corr.IMG
        geo_variable.attrs[corr.IMG_CORR_SCALE] = [-12, 0]
        geo_variable.attrs["pdf_shape"] = "gaussian"

    @staticmethod
    def _add_calibration_coeff_correlation_attributes(coeff_variable):
        coeff_variable.attrs[corr.SCAN_CORR_FORM] = corr.RECT
        coeff_variable.attrs[corr.SCAN_CORR_UNIT] = corr.PIXEL
        coeff_variable.attrs[corr.SCAN_CORR_SCALE] = [-np.inf, np.inf]
        coeff_variable.attrs[corr.TIME_CORR_FORM] = corr.RECT
        coeff_variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        coeff_variable.attrs[corr.TIME_CORR_SCALE] = [-np.inf, np.inf]
        coeff_variable.attrs[corr.IMG_CORR_FORM] = corr.TRI
        coeff_variable.attrs[corr.IMG_CORR_UNIT] = corr.MONTHS
        coeff_variable.attrs[corr.IMG_CORR_SCALE] = [-1.5, 1.5]
        coeff_variable.attrs["pdf_shape"] = "gaussian"

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

        tu.add_units(variable, "degree")
        tu.add_scale_factor(variable, scale_factor)
        return variable
