import numpy as np
from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil as tu

FILL_VALUE = -999.0
COUNTS_FILL_VALUE = 99999
NUM_CHANNELS = 19
NUM_RAD_CHANNELS = 20
NUM_COEFFS = 3
NUM_MINOR_FRAME = 23
SWATH_WIDTH = 56


class HIRS:
    @staticmethod
    def add_original_variables(dataset, height):
        tu.add_geolocation_variables(dataset, SWATH_WIDTH, height)

        # bt
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_CHANNELS, np.float32, FILL_VALUE)
        variable = Variable(["channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = FILL_VALUE
        variable.attrs["standard_name"] = "toa_brightness_temperature"
        variable.attrs["units"] = "K"
        variable.attrs["ancilliary_variables"] = "scnlinf qualind linqualflags chqualflags mnfrqualflags"
        dataset["bt"] = variable

        # c_earth
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_RAD_CHANNELS, np.int32,
                                                            COUNTS_FILL_VALUE, ["rad_channel", "y", "x"])
        variable = Variable(["rad_channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = COUNTS_FILL_VALUE
        variable.attrs["long_name"] = "counts_earth"
        variable.attrs["units"] = "count"
        tu.set_unsigned(variable)
        variable.attrs["ancilliary_variables"] = "scnlinf qualind linqualflags chqualflags mnfrqualflags"
        dataset["c_earth"] = variable

        # L_earth
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_RAD_CHANNELS, np.float32,
                                                            FILL_VALUE, ["rad_channel", "y", "x"])
        variable = Variable(["rad_channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = FILL_VALUE
        variable.attrs["standard_name"] = "toa_outgoing_inband_radiance"
        variable.attrs["units"] = "mW m^-2 sr^-1 cm"
        variable.attrs["ancilliary_variables"] = "scnlinf qualind linqualflags chqualflags mnfrqualflags"
        dataset["L_earth"] = variable

        dataset["sat_za"] = HIRS._create_geo_angle_variable("sensor_zenith_angle", height)
        dataset["sat_aa"] = HIRS._create_geo_angle_variable("sensor_azimuth_angle", height)
        dataset["sat_aa"].variable.attrs["long_name"] = "local_azimuth_angle"
        dataset["sol_za"] = HIRS._create_geo_angle_variable("solar_zenith_angle", height)
        dataset["sol_aa"] = HIRS._create_geo_angle_variable("solar_azimuth_angle", height)

        # scanline
        default_array = DefaultData.create_default_vector(height, np.int16)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["long_name"] = "scanline_number"
        variable.attrs["units"] = "count"
        dataset["scanline"] = variable

        # time
        default_array = DefaultData.create_default_vector(height, np.int32)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int32)
        tu.set_unsigned(variable)
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time in seconds since 1970-01-01 00:00:00"
        variable.attrs["units"] = "s"
        dataset["time"] = variable

        # scnlinf
        default_array = DefaultData.create_default_vector(height, np.int8, fill_value=9)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = 9
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "scanline_bitfield"
        variable.attrs["flag_values"] = "0, 1, 2, 3"
        variable.attrs["flag_meanings"] = "earth_view space_view icct_view iwct_view"
        dataset["scnlinf"] = variable

        # qualind
        default_array = DefaultData.create_default_vector(height, np.int32, fill_value=0)
        variable = Variable(["y"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "quality_indicator_bitfield"
        variable.attrs["flag_masks"] = "1, 2, 4, 8, 16, 32, 64, 128"
        variable.attrs[
            "flag_meanings"] = "do_not_use_scan time_sequence_error data_gap_preceding_scan no_calibration no_earth_location clock_update status_changed line_incomplete"
        dataset["qualind"] = variable

        # linqualflags
        default_array = DefaultData.create_default_vector(height, np.int32, fill_value=0)
        variable = Variable(["y"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "scanline_quality_flags_bitfield"
        variable.attrs[
            "flag_masks"] = "256, 512, 1024, 2048, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456"
        variable.attrs[
            "flag_meanings"] = "time_field_bad time_field_bad_not_inf inconsistent_sequence scan_time_repeat uncalib_bad_time calib_few_scans uncalib_bad_prt calib_marginal_prt uncalib_channels uncalib_inst_mode quest_ant_black_body zero_loc bad_loc_time bad_loc_marginal bad_loc_reason bad_loc_ant"
        dataset["linqualflags"] = variable

        # chqualflags
        default_array = DefaultData.create_default_array(NUM_CHANNELS, height, np.int32, dims_names=["y", "channel"],
                                                         fill_value=0)
        variable = Variable(["y", "channel"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "channel_quality_flags_bitfield"
        dataset["chqualflags"] = variable

        # mnfrqualflags
        default_array = DefaultData.create_default_array(NUM_MINOR_FRAME, height, np.int32,
                                                         dims_names=["y", "minor_frame"], fill_value=0)
        variable = Variable(["y", "minor_frame"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "minor_frame_quality_flags_bitfield"
        dataset["mnfrqualflags"] = variable

    @staticmethod
    def _create_geo_angle_variable(angle, height):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=FILL_VALUE)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = FILL_VALUE
        variable.attrs["standard_name"] = angle
        variable.attrs["units"] = "degree"
        return variable

    @staticmethod
    def get_swath_width():
        return SWATH_WIDTH

    @staticmethod
    def add_uncertainty_variables(dataset, height):
        # u_lat
        variable = HIRS._create_angle_variable(height, "uncertainty_latitude")
        dataset["u_lat"] = variable

        # u_lon
        variable = HIRS._create_angle_variable(height, "uncertainty_longitude")
        dataset["u_lon"] = variable

        # u_time
        variable = tu.create_float_variable(SWATH_WIDTH, height, "uncertainty_time")
        variable.attrs["units"] = "s"
        dataset["u_time"] = variable

        # u_c_earth
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_counts_Earth")
        variable.attrs["units"] = "count"
        dataset["u_c_earth"] = variable

        # u_L_earth_random
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_radiance_Earth_random")
        variable.attrs["units"] = "mW m^-2 sr^-1 cm"
        dataset["u_L_earth_random"] = variable

        # u_L_earth_structuredrandom
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_radiance_Earth_structured_random")
        variable.attrs["units"] = "mW m^-2 sr^-1 cm"
        dataset["u_L_earth_structuredrandom"] = variable

        # u_L_earth_systematic
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_radiance_Earth_systematic")
        variable.attrs["units"] = "mW m^-2 sr^-1 cm"
        dataset["u_L_earth_systematic"] = variable

        # u_L_earth_total
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_radiance_Earth_total")
        variable.attrs["units"] = "mW m^-2 sr^-1 cm"
        dataset["u_L_earth_total"] = variable

        # S_u_L_earth
        variable = tu.create_float_variable(NUM_RAD_CHANNELS, NUM_RAD_CHANNELS, "covariance_radiance_Earth",
                                            dim_names=["rad_channel", "rad_channel"])
        dataset["S_u_L_earth"] = variable

        # u_bt_random
        variable = HIRS._create_3d_bt_uncertainty_variable(height, "uncertainty_bt_random")
        variable.attrs["units"] = "K"
        dataset["u_bt_random"] = variable

        # u_bt_structuredrandom
        variable = HIRS._create_3d_bt_uncertainty_variable(height, "uncertainty_bt_structured_random")
        variable.attrs["units"] = "K"
        dataset["u_bt_structuredrandom"] = variable

        # u_bt_systematic
        variable = HIRS._create_3d_bt_uncertainty_variable(height, "uncertainty_bt_systematic")
        variable.attrs["units"] = "K"
        dataset["u_bt_systematic"] = variable

        # u_bt_total
        variable = HIRS._create_3d_bt_uncertainty_variable(height, "uncertainty_bt_total")
        variable.attrs["units"] = "K"
        dataset["u_bt_total"] = variable

        # S_bt
        variable = tu.create_float_variable(NUM_CHANNELS, NUM_CHANNELS, "covariance_brightness_temperature",
                                            dim_names=["channel", "channel"])
        dataset["S_bt"] = variable

        # calcof
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_COEFFS, np.float32,
                                                            dims_names=["coeffs", "y", "x"])
        variable = Variable(["coeffs", "y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "calibration_coefficients"
        dataset["calcof"] = variable

        # u_calcof
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_COEFFS, np.float32,
                                                            dims_names=["coeffs", "y", "x"])
        variable = Variable(["coeffs", "y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "uncertainty_calibration_coefficients"
        dataset["u_calcof"] = variable

        dataset["Tc_baseplate"] = HIRS._create_counts_vector(height, "temperature_baseplate_counts")
        dataset["Tc_ch"] = HIRS._create_counts_vector(height, "temperature_coolerhousing_counts")
        dataset["Tc_elec"] = HIRS._create_counts_vector(height, "temperature_electronics_counts")
        dataset["Tc_fsr"] = HIRS._create_counts_vector(height, "temperature_first_stage_radiator_counts")
        dataset["Tc_fwh"] = HIRS._create_counts_vector(height, "temperature_filter_wheel_housing_counts")
        dataset["Tc_fwm"] = HIRS._create_counts_vector(height, "temperature_filter_wheel_monitor_counts")
        dataset["Tc_icct"] = HIRS._create_counts_vector(height, "temperature_internal_cold_calibration_target_counts")
        dataset["Tc_iwct"] = HIRS._create_counts_vector(height, "temperature_internal_warm_calibration_target_counts")
        dataset["Tc_patch_exp"] = HIRS._create_counts_vector(height, "temperature_patch_expanded_scale_counts")
        dataset["Tc_patch_full"] = HIRS._create_counts_vector(height, "temperature_patch_full_range_counts")
        dataset["Tc_tlscp_prim"] = HIRS._create_counts_vector(height, "temperature_telescope_primary_counts")
        dataset["Tc_tlscp_sec"] = HIRS._create_counts_vector(height, "temperature_telescope_secondary_counts")
        dataset["Tc_tlscp_tert"] = HIRS._create_counts_vector(height, "temperature_telescope_tertiary_counts")
        dataset["Tc_scanmirror"] = HIRS._create_counts_vector(height, "temperature_scanmirror_counts")
        dataset["Tc_scanmotor"] = HIRS._create_counts_vector(height, "temperature_scanmotor_counts")

        dataset["u_Tc_baseplate"] = HIRS._create_counts_uncertainty_vector(height,
                                                                           "uncertainty_temperature_baseplate_counts")
        dataset["u_Tc_ch"] = HIRS._create_counts_uncertainty_vector(height,
                                                                    "uncertainty_temperature_coolerhousing_counts")
        dataset["u_Tc_elec"] = HIRS._create_counts_uncertainty_vector(height,
                                                                      "uncertainty_temperature_electronics_counts")
        dataset["u_Tc_fsr"] = HIRS._create_counts_uncertainty_vector(height,
                                                                     "uncertainty_temperature_first_stage_radiator_counts")
        dataset["u_Tc_fwh"] = HIRS._create_counts_uncertainty_vector(height,
                                                                     "uncertainty_temperature_filter_wheel_housing_counts")
        dataset["u_Tc_fwm"] = HIRS._create_counts_uncertainty_vector(height,
                                                                     "uncertainty_temperature_filter_wheel_monitor_counts")
        dataset["u_Tc_icct"] = HIRS._create_counts_uncertainty_vector(height,
                                                                      "uncertainty_temperature_internal_cold_calibration_target_counts")
        dataset["u_Tc_iwct"] = HIRS._create_counts_uncertainty_vector(height,
                                                                      "uncertainty_temperature_internal_warm_calibration_target_counts")
        dataset["u_Tc_patch_exp"] = HIRS._create_counts_uncertainty_vector(height,
                                                                           "uncertainty_temperature_patch_expanded_scale_counts")
        dataset["u_Tc_patch_full"] = HIRS._create_counts_uncertainty_vector(height,
                                                                            "uncertainty_temperature_patch_full_range_counts")
        dataset["u_Tc_tlscp_prim"] = HIRS._create_counts_uncertainty_vector(height,
                                                                            "uncertainty_temperature_telescope_primary_counts")
        dataset["u_Tc_tlscp_sec"] = HIRS._create_counts_uncertainty_vector(height,
                                                                           "uncertainty_temperature_telescope_secondary_counts")
        dataset["u_Tc_tlscp_tert"] = HIRS._create_counts_uncertainty_vector(height,
                                                                            "uncertainty_temperature_telescope_tertiary_counts")
        dataset["u_Tc_scanmirror"] = HIRS._create_counts_uncertainty_vector(height,
                                                                            "uncertainty_temperature_scanmirror_counts")
        dataset["u_Tc_scanmotor"] = HIRS._create_counts_uncertainty_vector(height,
                                                                           "uncertainty_temperature_scanmotor_counts")

        dataset["TK_baseplate"] = HIRS._create_temperature_vector(height, "temperature_baseplate_K")
        dataset["TK_ch"] = HIRS._create_temperature_vector(height, "temperature_coolerhousing_K")
        dataset["TK_elec"] = HIRS._create_temperature_vector(height, "temperature_electronics_K")
        dataset["TK_fsr"] = HIRS._create_temperature_vector(height, "temperature_first_stage_radiator_K")
        dataset["TK_fwh"] = HIRS._create_temperature_vector(height, "temperature_filter_wheel_housing_K")
        dataset["TK_fwm"] = HIRS._create_temperature_vector(height, "temperature_filter_wheel_monitor_K")
        dataset["TK_icct"] = HIRS._create_temperature_vector(height, "temperature_internal_cold_calibration_target_K")
        dataset["TK_iwct"] = HIRS._create_temperature_vector(height, "temperature_internal_warm_calibration_target_K")
        dataset["TK_patch_exp"] = HIRS._create_temperature_vector(height, "temperature_patch_expanded_scale_K")
        dataset["TK_patch_full"] = HIRS._create_temperature_vector(height, "temperature_patch_full_range_K")
        dataset["TK_tlscp_prim"] = HIRS._create_temperature_vector(height, "temperature_telescope_primary_K")
        dataset["TK_tlscp_sec"] = HIRS._create_temperature_vector(height, "temperature_telescope_secondary_K")
        dataset["TK_tlscp_tert"] = HIRS._create_temperature_vector(height, "temperature_telescope_tertiary_K")
        dataset["TK_scanmirror"] = HIRS._create_temperature_vector(height, "temperature_scanmirror_K")
        dataset["TK_scanmotor"] = HIRS._create_temperature_vector(height, "temperature_scanmotor_K")

        dataset["u_TK_baseplate"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_baseplate_K")
        dataset["u_TK_ch"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_coolerhousing_K")
        dataset["u_TK_elec"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_electronics_K")
        dataset["u_TK_fsr"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_first_stage_radiator_K")
        dataset["u_TK_fwh"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_filter_wheel_housing_K")
        dataset["u_TK_fwm"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_filter_wheel_monitor_K")
        dataset["u_TK_icct"] = HIRS._create_temperature_vector(height,
                                                               "uncertainty_temperature_internal_cold_calibration_target_K")
        dataset["u_TK_iwct"] = HIRS._create_temperature_vector(height,
                                                               "uncertainty_temperature_internal_warm_calibration_target_K")
        dataset["u_TK_patch_exp"] = HIRS._create_temperature_vector(height,
                                                                    "uncertainty_temperature_patch_expanded_scale_K")
        dataset["u_TK_patch_full"] = HIRS._create_temperature_vector(height,
                                                                     "uncertainty_temperature_patch_full_range_K")
        dataset["u_TK_tlscp_prim"] = HIRS._create_temperature_vector(height,
                                                                     "uncertainty_temperature_telescope_primary_K")
        dataset["u_TK_tlscp_sec"] = HIRS._create_temperature_vector(height,
                                                                    "uncertainty_temperature_telescope_secondary_K")
        dataset["u_TK_tlscp_tert"] = HIRS._create_temperature_vector(height,
                                                                     "uncertainty_temperature_telescope_tertiary_K")
        dataset["u_TK_scanmirror"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_scanmirror_K")
        dataset["u_TK_scanmotor"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_scanmotor_K")

        dataset["u_sol_za"] = HIRS._create_geo_angle_variable("uncertainty_solar_zenith_angle", height)
        dataset["u_sol_aa"] = HIRS._create_geo_angle_variable("uncertainty_solar_azimuth_angle", height)
        dataset["u_sat_za"] = HIRS._create_geo_angle_variable("uncertainty_satellite_zenith_angle", height)
        dataset["u_sat_aa"] = HIRS._create_geo_angle_variable("uncertainty_local_azimuth_angle", height)

    @staticmethod
    def _create_temperature_vector(height, standard_name):
        default_array = DefaultData.create_default_vector(height, np.float32)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = standard_name
        variable.attrs["units"] = "K"
        return variable

    @staticmethod
    def _create_counts_vector(height, standard_name):
        default_array = DefaultData.create_default_vector(height, np.int32)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int32)
        variable.attrs["standard_name"] = standard_name
        variable.attrs["units"] = "count"
        return variable

    @staticmethod
    def _create_counts_uncertainty_vector(height, standard_name):
        default_array = DefaultData.create_default_vector(height, np.float32)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = standard_name
        variable.attrs["units"] = "count"
        return variable

    @staticmethod
    def _create_3d_rad_uncertainty_variable(height, standard_name):
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_RAD_CHANNELS, np.float32,
                                                            dims_names=["rad_channel", "y", "x"])
        variable = Variable(["rad_channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = standard_name
        return variable

    @staticmethod
    def _create_3d_bt_uncertainty_variable(height, standard_name):
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_CHANNELS, np.float32,
                                                            dims_names=["channel", "y", "x"])
        variable = Variable(["channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = standard_name
        return variable

    @staticmethod
    def _create_angle_variable(height, standard_name):
        variable = tu.create_float_variable(SWATH_WIDTH, height, standard_name)
        variable.attrs["units"] = "degree"
        return variable
