import numpy as np
from xarray import Variable

from writer.correlation import Correlation as corr
from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil as tu

FULL_DIMENSION = 5000
IR_DIMENSION = 2500
SRF_SIZE = 1011
SOL_IRR_SIZE = 24

TIME_FILL_VALUE = -32768


class MVIRI:
    @staticmethod
    def add_original_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05
        # time
        default_array = DefaultData.create_default_array(IR_DIMENSION, IR_DIMENSION, np.uint16)
        variable = Variable(["y_ir", "x_ir"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time of pixel"
        tu.add_units(variable, "seconds since 1970-01-01 00:00:00")
        tu.add_offset(variable, TIME_FILL_VALUE)
        dataset["time"] = variable

        dataset["satellite_azimuth_angle"] = MVIRI._create_angle_variable_int(0.005493164, standard_name="sensor_azimuth_angle", unsigned=True)
        dataset["satellite_zenith_angle"] = MVIRI._create_angle_variable_int(0.005493248, standard_name="sensor_zenith_angle")
        dataset["solar_azimuth_angle"] = MVIRI._create_angle_variable_int(0.005493164, standard_name="solar_azimuth_angle", unsigned=True)
        dataset["solar_zenith_angle"] = MVIRI._create_angle_variable_int(0.005493248, standard_name="solar_zenith_angle")

        # count_ir
        default_array = DefaultData.create_default_array(IR_DIMENSION, IR_DIMENSION, np.uint8)
        variable = Variable(["y_ir", "x_ir"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint8))
        variable.attrs["long_name"] = "Infrared Image Counts"
        tu.add_units(variable, "count")
        dataset["count_ir"] = variable

        # count_wv
        default_array = DefaultData.create_default_array(IR_DIMENSION, IR_DIMENSION, np.uint8)
        variable = Variable(["y_ir", "x_ir"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint8))
        variable.attrs["long_name"] = "WV Image Counts"
        tu.add_units(variable, "count")
        dataset["count_wv"] = variable

        # distance_sun_earth
        dataset["distance_sun_earth"] = tu.create_scalar_float_variable(long_name="Sun-Earth distance", units="au")

        # sol_eff_irr
        dataset["sol_eff_irr"] = tu.create_scalar_float_variable(standard_name="solar_irradiance_vis", long_name="Solar effective Irradiance", units="W*m-2")

        # srf
        default_array = DefaultData.create_default_vector(SRF_SIZE, np.float32)
        variable = Variable(["srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Spectral Response Function"
        dataset["spectral_response_function_vis"] = variable

        # srf covariance_
        default_array = DefaultData.create_default_array(SRF_SIZE, SRF_SIZE, np.float32)
        variable = Variable(["srf_size", "srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Covariance of the Visible Band Spectral Response Function"
        dataset["covariance_spectral_response_function_vis"] = variable

        dataset["a_ir"] = tu.create_scalar_float_variable(long_name="Calibration parameter a for IR Band", units="mWm^-2sr^-1cm^-1")
        dataset["b_ir"] = tu.create_scalar_float_variable(long_name="Calibration parameter b for IR Band", units="mWm^-2sr^-1cm^-1/DC")
        dataset["u_a_ir"] = tu.create_scalar_float_variable(long_name="Uncertainty of calibration parameter a for IR Band", units="mWm^-2sr^-1cm^-1")
        dataset["u_b_ir"] = tu.create_scalar_float_variable(long_name="Uncertainty of calibration parameter b for IR Band", units="mWm^-2sr^-1cm^-1/DC")
        dataset["q_wv"] = tu.create_scalar_float_variable(long_name="WV Band Calibration quality flag", units="1")
        dataset["unit_conversion_ir"] = tu.create_scalar_float_variable(long_name="IR Unit conversion factor", units="1")
        dataset["unit_conversion_wv"] = tu.create_scalar_float_variable(long_name="WV Unit conversion factor", units="1")
        dataset["bt_a_ir"] = tu.create_scalar_float_variable(long_name="IR Band BT conversion parameter A", units="1")
        dataset["bt_b_ir"] = tu.create_scalar_float_variable(long_name="IR Band BT conversion parameter B", units="1")
        dataset["bt_a_wv"] = tu.create_scalar_float_variable(long_name="WV Band BT conversion parameter A", units="1")
        dataset["bt_b_wv"] = tu.create_scalar_float_variable(long_name="WV Band BT conversion parameter B", units="1")

    @staticmethod
    def get_swath_width():
        return FULL_DIMENSION

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05

        # reflectance
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        variable.attrs["standard_name"] = "toa_bidirectional_reflectance"
        tu.add_units(variable, "percent")
        tu.add_scale_factor(variable, 1.52588E-05)
        dataset["reflectance"] = variable

        # u_random
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["long_name"] = "random uncertainty per pixel"
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        tu.add_units(variable, "percent")
        tu.add_scale_factor(variable, 1.52588E-05)
        dataset["u_random"] = variable

        # u_non_random
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["long_name"] = "non-random uncertainty per pixel"
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        tu.add_units(variable, "percent")
        tu.add_scale_factor(variable, 1.52588E-05)
        dataset["u_non_random"] = variable

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-02-05

        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        tu.add_units(variable, "percent")
        tu.add_scale_factor(variable, 1e-03)
        variable.attrs[corr.SCAN_CORR_FORM] = corr.TRUNC_GAUSS
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.PIXEL
        variable.attrs[corr.SCAN_CORR_SCALE] = [-2, 2]
        variable.attrs[corr.TIME_CORR_FORM] = corr.TRUNC_GAUSS
        variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        variable.attrs[corr.TIME_CORR_SCALE] = [-2, 2]
        variable.attrs[corr.IMG_CORR_FORM] = corr.TRI
        variable.attrs[corr.IMG_CORR_UNIT] = corr.MONTHS
        variable.attrs[corr.IMG_CORR_SCALE] = [-3, 3]
        variable.attrs["pdf_shape"] = "gaussian"
        dataset["u_toa_bidirectional_reflectance_vis"] = variable

        # count_vis
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint8)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint8))
        variable.attrs["long_name"] = "Image counts"
        tu.add_units(variable, "count")
        dataset["count_vis"] = variable

        dataset["u_latitude"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Latitude", unsigned=True)
        MVIRI._add_geo_correlation_attributes(dataset["u_latitude"])

        dataset["u_longitude"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Longitude", unsigned=True)
        MVIRI._add_geo_correlation_attributes(dataset["u_longitude"])

        # u_time
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        variable.attrs["standard_name"] = "Uncertainty in Time"
        tu.add_units(variable, "s")
        tu.add_scale_factor(variable, 0.009155273)
        variable.attrs["pdf_shape"] = "rectangle"
        dataset["u_time"] = variable

        dataset["u_satellite_zenith_angle"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Satellite Zenith Angle", unsigned=True)
        dataset["u_satellite_azimuth_angle"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Satellite Azimuth Angle", unsigned=True)
        dataset["u_solar_zenith_angle"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Solar Zenith Angle", unsigned=True)
        dataset["u_solar_azimuth_angle"] = MVIRI._create_angle_variable_int(7.62939E-05, long_name="Uncertainty in Solar Azimuth Angle", unsigned=True)

        # u_combined_counts_vis
        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, np.uint16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        variable.attrs["long_name"] = "Total Uncertainty in counts"
        tu.add_units(variable, "count")
        tu.add_scale_factor(variable, 7.62939E-05)
        variable.attrs[corr.SCAN_CORR_FORM] = corr.EIFFEL
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.PIXEL
        variable.attrs[corr.SCAN_CORR_SCALE] = [-2, 2]
        variable.attrs[corr.TIME_CORR_FORM] = corr.EIFFEL
        variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        variable.attrs[corr.TIME_CORR_SCALE] = [-2, 2]
        variable.attrs["pdf_shape"] = "digitised_gaussian"
        dataset["u_combined_counts_vis"] = variable

        # u_srf
        default_array = DefaultData.create_default_array(SRF_SIZE, SRF_SIZE, np.uint16)
        variable = Variable(["srf_size", "srf_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        variable.attrs["long_name"] = "Uncertainty in SRF"
        tu.add_scale_factor(variable, 1.52588E-05)
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

        dataset["a0_vis"] = tu.create_scalar_float_variable("Calibration Coefficient at Launch", units="Wm^-2sr^-1/count")
        dataset["a1_vis"] = tu.create_scalar_float_variable("Time variation of a0", units="Wm^-2sr^-1/count day^-1 10^5")
        dataset["mean_counts_space_vis"] = tu.create_scalar_float_variable("Space count", units="count")

        # u_a0_vis
        variable = tu.create_scalar_float_variable("Uncertainty in a0", units="Wm^-2sr^-1/count")
        MVIRI._add_calibration_coeff_correlation_attributes(variable)
        dataset["u_a0_vis"] = variable

        # u_a1_vis
        variable = tu.create_scalar_float_variable("Uncertainty in a1", units="Wm^-2sr^-1/count day^-1 10^5")
        MVIRI._add_calibration_coeff_correlation_attributes(variable)
        dataset["u_a1_vis"] = variable

        # covariance_a0_a1_vis
        default_array = DefaultData.create_default_array(2, 2, np.float32)
        variable = Variable(["calib_coeff_cov_size", "calib_coeff_cov_size"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Covariance matrix of calibration coefficients"
        dataset["covariance_a0_a1_vis"] = variable

        # u_sol_eff_irr
        default_array = DefaultData.get_default_fill_value(np.float32)
        variable = Variable([], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["long_name"] = "Uncertainty in Solar effective Irradiance"
        tu.add_units(variable, "Wm^-2")
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

        dataset["u_electronics_counts_vis"] = tu.create_scalar_float_variable("Uncertainty due to Electronics noise", units="count")
        dataset["u_digitization_counts_vis"] = tu.create_scalar_float_variable("Uncertainty due to digitization", units="count")

        # allan_deviation_counts_space_vis
        variable = tu.create_scalar_float_variable("Uncertainty of space count", units="count")
        variable.attrs[corr.TIME_CORR_FORM] = corr.RECT
        variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        variable.attrs[corr.TIME_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs["pdf_shape"] = "digitised_gaussian"
        dataset["allan_deviation_counts_space_vis"] = variable

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
    def _create_angle_variable_int(scale_factor, standard_name=None, long_name=None, unsigned=False, fill_value=None):
        if unsigned is True:
            data_type = np.uint16
        else:
            data_type=np.int16

        default_array = DefaultData.create_default_array(FULL_DIMENSION, FULL_DIMENSION, data_type, fill_value=fill_value)
        variable = Variable(["y", "x"], default_array)

        if fill_value is not None:
            tu.add_fill_value(variable, fill_value)
        else:
            tu.add_fill_value(variable, DefaultData.get_default_fill_value(data_type))

        if standard_name is not None:
            variable.attrs["standard_name"] = standard_name

        if long_name is not None:
            variable.attrs["long_name"] = long_name

        tu.add_units(variable, "degree")
        tu.add_scale_factor(variable, scale_factor)
        return variable
