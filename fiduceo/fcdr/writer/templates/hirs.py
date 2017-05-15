import numpy as np
from xarray import Variable

from fiduceo.fcdr.writer.correlation import Correlation as corr
from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.templateutil import TemplateUtil as tu

FILL_VALUE = -999.0
COUNTS_FILL_VALUE = 99999
NUM_CHANNELS = 19
NUM_RAD_CHANNELS = 20
NUM_COEFFS = 3
NUM_MINOR_FRAME = 64
NUM_CALIBRATION_CYCLE = 337
NUM_SCAN_ANGLES = 168
PRT_NUMBER = 4
PRT_NUMBER_IWT = 4
PRT_READING = 5
SWATH_WIDTH = 56
WIDTH_TODO = 60


class HIRS:
    @staticmethod
    def add_geolocation_variables(dataset, height):
        tu.add_geolocation_variables(dataset, SWATH_WIDTH, height)

    @staticmethod
    def add_extended_flag_variables(dataset, height):
        # linqualflags
        default_array = DefaultData.create_default_vector(height, np.int32, fill_value=0)
        variable = Variable(["y"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "scanline_quality_flags_bitfield"
        variable.attrs["flag_masks"] = "256, 512, 1024, 2048, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456"
        variable.attrs[
            "flag_meanings"] = "time_field_bad time_field_bad_not_inf inconsistent_sequence scan_time_repeat uncalib_bad_time calib_few_scans uncalib_bad_prt calib_marginal_prt uncalib_channels uncalib_inst_mode quest_ant_black_body zero_loc bad_loc_time bad_loc_marginal bad_loc_reason bad_loc_ant"
        dataset["linqualflags"] = variable
        # chqualflags
        default_array = DefaultData.create_default_array(NUM_CHANNELS, height, np.int32, dims_names=["y", "channel"], fill_value=0)
        variable = Variable(["y", "channel"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "channel_quality_flags_bitfield"
        dataset["chqualflags"] = variable
        # mnfrqualflags
        default_array = DefaultData.create_default_array(NUM_MINOR_FRAME, height, np.int32, dims_names=["y", "minor_frame"], fill_value=0)
        variable = Variable(["y", "minor_frame"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "minor_frame_quality_flags_bitfield"
        dataset["mnfrqualflags"] = variable

    @staticmethod
    def add_common_sensor_variables(dataset, height):
        # scanline
        default_array = DefaultData.create_default_vector(height, np.int16)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "scanline_number"
        tu.add_units(variable, "count")
        dataset["scanline"] = variable
        # time
        default_array = DefaultData.create_default_vector(height, np.uint32)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint32))
        variable.attrs["standard_name"] = "time"
        variable.attrs["long_name"] = "Acquisition time in seconds since 1970-01-01 00:00:00"
        tu.add_units(variable, "s")
        dataset["time"] = variable
        # scnlintime
        variable = HIRS._create_int32_vector(height, "time")
        variable.attrs["long_name"] = "Scan line time of day"
        variable.attrs["orig_name"] = "hrs_scnlintime"
        tu.add_units(variable, "ms")
        dataset["scnlintime"] = variable
        # scnlinf
        default_array = DefaultData.create_default_vector(height, np.int16, fill_value=0)
        variable = Variable(["y"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "scanline_bitfield"
        variable.attrs["flag_masks"] = "16384, 32768"
        variable.attrs["flag_meanings"] = "clock_drift_correction southbound_data"
        dataset["scnlinf"] = variable
        # scantype
        default_array = DefaultData.create_default_vector(height, np.int8, fill_value=0)
        variable = Variable(["y"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "scantype_bitfield"
        variable.attrs["flag_values"] = "0, 1, 2, 3"
        variable.attrs["flag_meanings"] = "earth_view space_view cold_bb_view main_bb_view"
        dataset["scantype"] = variable
        # qualind
        default_array = DefaultData.create_default_vector(height, np.int32, fill_value=0)
        variable = Variable(["y"], default_array)
        variable.attrs["standard_name"] = "status_flag"
        variable.attrs["long_name"] = "quality_indicator_bitfield"
        variable.attrs["flag_masks"] = "1, 2, 4, 8, 16, 32, 64, 128"
        variable.attrs["flag_meanings"] = "do_not_use_scan time_sequence_error data_gap_preceding_scan no_calibration no_earth_location clock_update status_changed line_incomplete"
        dataset["qualind"] = variable

    @staticmethod
    def add_common_angles(dataset, height):
        dataset["satellite_zenith_angle"] = HIRS._create_geo_angle_variable("platform_zenith_angle", height)
        dataset["satellite_azimuth_angle"] = HIRS._create_geo_angle_variable("sensor_azimuth_angle", height)
        dataset["satellite_azimuth_angle"].variable.attrs["long_name"] = "local_azimuth_angle"

        dataset["solar_zenith_angle"] = HIRS._create_geo_angle_variable("solar_zenith_angle", height, orig_name="solar_zenith_angle")
        dataset["solar_azimuth_angle"] = HIRS._create_geo_angle_variable("solar_azimuth_angle", height)

    @staticmethod
    def add_bt_variable(dataset, height):
        # bt
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_CHANNELS, np.float32, np.NaN)
        variable = Variable(["channel", "y", "x"], default_array)
        variable.attrs["standard_name"] = "toa_brightness_temperature"
        variable.attrs["long_name"] = "Brightness temperature, NOAA/EUMETSAT calibrated"
        tu.add_units(variable, "K")
        tu.add_encoding(variable, np.int16, FILL_VALUE, 0.01, 150.0)
        variable.attrs["ancilliary_variables"] = "scnlinf scantype qualind linqualflags chqualflags mnfrqualflags"
        dataset["bt"] = variable

    @staticmethod
    def get_swath_width():
        return SWATH_WIDTH

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_CHANNELS, np.float32, np.NaN)
        variable = Variable(["channel", "y", "x"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "random uncertainty per pixel"
        tu.add_units(variable, "percent")
        dataset["u_random"] = variable

        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_CHANNELS, np.float32, np.NaN)
        variable = Variable(["channel", "y", "x"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "non-random uncertainty per pixel"
        tu.add_units(variable, "percent")
        dataset["u_non_random"] = variable

    @staticmethod
    def add_common_full_fcdr_variables(dataset, height):
        # @todo 1 tb/tb continue here 2017-05-09
        pass

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        # c_earth
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_RAD_CHANNELS, np.uint16, dims_names=["rad_channel", "y", "x"])
        variable = Variable(["rad_channel", "y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        variable.attrs["long_name"] = "counts_earth"
        tu.add_units(variable, "count")
        variable.attrs["ancilliary_variables"] = "scnlinf qualind linqualflags chqualflags mnfrqualflags"
        dataset["c_earth"] = variable

        # L_earth
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_RAD_CHANNELS, np.float32, np.NaN, ["rad_channel", "y", "x"])
        variable = Variable(["rad_channel", "y", "x"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["standard_name"] = "toa_outgoing_inband_radiance"
        tu.add_units(variable, "W/Hz/m ** 2/sr")
        variable.attrs["long_name"] = "Channel radiance, NOAA/EUMETSAT calibrated"
        variable.attrs["orig_name"] = "radiance"
        variable.attrs["ancilliary_variables"] = "scnlinf qualind linqualflags chqualflags mnfrqualflags"
        dataset["L_earth"] = variable

        # u_lat
        variable = HIRS._create_angle_variable(height, "uncertainty_latitude")
        dataset["u_lat"] = variable

        # u_lon
        variable = HIRS._create_angle_variable(height, "uncertainty_longitude")
        dataset["u_lon"] = variable

        # u_time
        variable = tu.create_float_variable(SWATH_WIDTH, height, "uncertainty_time")
        tu.add_units(variable, "s")
        dataset["u_time"] = variable

        # u_c_earth
        default_array = DefaultData.create_default_array(NUM_CALIBRATION_CYCLE, NUM_CHANNELS, np.uint16, dims_names=["channel", "calibration_cycle"])
        variable = Variable(["channel", "calibration_cycle"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        tu.add_units(variable, "count")
        tu.add_scale_factor(variable, 0.005)
        variable.attrs["long_name"] = "uncertainty counts for Earth views"
        variable.attrs["ancilliary_variables"] = "u_c_earth_chan_corr"
        variable.attrs["channels_affected"] = "all"
        variable.attrs["parameter"] = "C_E"
        variable.attrs["pdf_shape"] = "gaussian"
        dataset["u_c_earth"] = variable

        # u_L_earth_random
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_radiance_Earth_random")
        tu.add_units(variable, "mW m^-2 sr^-1 cm")
        dataset["u_L_earth_random"] = variable

        # u_L_earth_structuredrandom
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_radiance_Earth_structured_random")
        tu.add_units(variable, "mW m^-2 sr^-1 cm")
        dataset["u_L_earth_structuredrandom"] = variable

        # u_L_earth_systematic
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_radiance_Earth_systematic")
        tu.add_units(variable, "mW m^-2 sr^-1 cm")
        dataset["u_L_earth_systematic"] = variable

        # u_L_earth_total
        variable = HIRS._create_3d_rad_uncertainty_variable(height, "uncertainty_radiance_Earth_total")
        tu.add_units(variable, "mW m^-2 sr^-1 cm")
        dataset["u_L_earth_total"] = variable

        # S_u_L_earth
        variable = tu.create_float_variable(NUM_RAD_CHANNELS, NUM_RAD_CHANNELS, "covariance_radiance_Earth", dim_names=["rad_channel", "rad_channel"])
        dataset["S_u_L_earth"] = variable

        # u_bt_random
        variable = HIRS._create_3d_bt_uncertainty_variable(height, "uncertainty_bt_random")
        tu.add_units(variable, "K")
        dataset["u_bt_random"] = variable

        # u_bt_structuredrandom
        variable = HIRS._create_3d_bt_uncertainty_variable(height, "uncertainty_bt_structured_random")
        tu.add_units(variable, "K")
        dataset["u_bt_structuredrandom"] = variable

        # u_bt_systematic
        variable = HIRS._create_3d_bt_uncertainty_variable(height, "uncertainty_bt_systematic")
        tu.add_units(variable, "K")
        dataset["u_bt_systematic"] = variable

        # u_bt_total
        variable = HIRS._create_3d_bt_uncertainty_variable(height, "uncertainty_bt_total")
        tu.add_units(variable, "K")
        dataset["u_bt_total"] = variable

        # S_bt
        variable = tu.create_float_variable(NUM_CHANNELS, NUM_CHANNELS, "covariance_brightness_temperature", dim_names=["channel", "channel"])
        dataset["S_bt"] = variable

        # calcof
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_COEFFS, np.float32, dims_names=["coeffs", "y", "x"])
        variable = Variable(["coeffs", "y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "calibration_coefficients"
        dataset["calcof"] = variable

        # u_calcof
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_COEFFS, np.float32, dims_names=["coeffs", "y", "x"])
        variable = Variable(["coeffs", "y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "uncertainty_calibration_coefficients"
        dataset["u_calcof"] = variable

        # navigation_status
        variable = HIRS._create_int32_vector(height, long_name="Navigation status bit field", orig_name="hrs_navstat")
        variable.attrs["standard_name"] = "status_flag"
        dataset["navigation_status"] = variable

        # quality_flags
        variable = HIRS._create_int32_vector(height, long_name="Quality indicator bit field", orig_name="hrs_qualind")
        variable.attrs["standard_name"] = "status_flag"
        dataset["quality_flags"] = variable

        variable = HIRS._create_float32_vector(np.NaN, height, "Platform altitude", "hrs_scalti")
        tu.add_units(variable, "km")
        dataset["platform_altitude"] = variable

        dataset["platform_pitch_angle"] = HIRS._create_float32_angle_vector(np.NaN, height, "Platform pitch angle", "hrs_pitchang")
        dataset["platform_roll_angle"] = HIRS._create_float32_angle_vector(np.NaN, height, "Platform roll angle", "hrs_rollang")
        dataset["platform_yaw_angle"] = HIRS._create_float32_angle_vector(np.NaN, height, "Platform yaw angle", "hrs_yawang")

        # scan_angles
        default_array = DefaultData.create_default_array(NUM_SCAN_ANGLES, height, np.float32, dims_names=["y", "num_scan_angles"], fill_value=np.NaN)
        variable = Variable(["y", "num_scan_angles"], default_array)
        tu.add_fill_value(variable, np.NaN)
        tu.add_units(variable, "degree")
        variable.attrs["long_name"] = "Scan angles"
        variable.attrs["orig_name"] = "hrs_ang"
        dataset["scan_angles"] = variable

        dataset["scanline_number"] = HIRS._create_int32_vector(height, long_name="scanline number", orig_name="hrs_scnlin")
        dataset["scanline_position"] = HIRS._create_int32_vector(height, long_name="Scanline position number in 32 second cycle", orig_name="hrs_scnpos")

        # second_original_calibration_coefficients
        default_array = DefaultData.create_default_array(WIDTH_TODO, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "width_todo"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Second original calibration coefficients (unsorted)"
        variable.attrs["orig_name"] = "hrs_scalcof"
        dataset["second_original_calibration_coefficients"] = variable

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

        dataset["u_Tc_baseplate"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_baseplate_counts")
        dataset["u_Tc_ch"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_coolerhousing_counts")
        dataset["u_Tc_elec"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_electronics_counts")
        dataset["u_Tc_fsr"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_first_stage_radiator_counts")
        dataset["u_Tc_fwh"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_filter_wheel_housing_counts")
        dataset["u_Tc_fwm"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_filter_wheel_monitor_counts")
        dataset["u_Tc_icct"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_internal_cold_calibration_target_counts")
        dataset["u_Tc_iwct"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_internal_warm_calibration_target_counts")
        dataset["u_Tc_patch_exp"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_patch_expanded_scale_counts")
        dataset["u_Tc_patch_full"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_patch_full_range_counts")
        dataset["u_Tc_tlscp_prim"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_telescope_primary_counts")
        dataset["u_Tc_tlscp_sec"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_telescope_secondary_counts")
        dataset["u_Tc_tlscp_tert"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_telescope_tertiary_counts")
        dataset["u_Tc_scanmirror"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_scanmirror_counts")
        dataset["u_Tc_scanmotor"] = HIRS._create_counts_uncertainty_vector(height, "uncertainty_temperature_scanmotor_counts")

        dataset["TK_baseplate"] = HIRS._create_temperature_vector(height, "Temperature baseplate", "temp_baseplate", np.NaN)
        dataset["TK_baseplate_analog"] = HIRS._create_temperature_vector(height, "Temperature baseplate (analog)", "temp_an_baseplate", np.NaN)
        dataset["TK_ch"] = HIRS._create_temperature_vector(height, "Temperature cooler housing", "temp_ch", np.NaN)
        dataset["TK_elec"] = HIRS._create_temperature_vector(height, "Temperature electronics", "temp_elec", np.NaN)
        dataset["TK_elec_analog"] = HIRS._create_temperature_vector(height, "Temperature electronics (analog)", "temp_an_el", np.NaN)
        dataset["TK_radiator_analog"] = HIRS._create_temperature_vector(height, "temperature_radiator_analog_K", "temp_an_rd", np.NaN)

        default_array = DefaultData.create_default_array(PRT_READING, height, np.float32, dims_names=["y", "prt_reading"], fill_value=np.NaN)
        variable = Variable(["y", "prt_reading"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Temperature first stage radiator"
        variable.attrs["orig_name"] = "temp_fsr"
        tu.add_units(variable, "K")
        dataset["TK_fsr"] = variable

        dataset["TK_fwh"] = HIRS._create_temperature_array_3d(height, "Temperature filter wheel housing", "temp_fwh", ["prt_number", "y", "prt_reading"])
        dataset["TK_fwm"] = HIRS._create_temperature_vector(height, "Temperature filter wheel motor", "temp_fwm", np.NaN)
        dataset["TK_fwm_analog"] = HIRS._create_temperature_vector(height, "Temperature filter wheel motor (analogue)", "temp_an_fwm", np.NaN)
        dataset["TK_icct"] = HIRS._create_temperature_vector(height, "temperature_internal_cold_calibration_target_K")
        dataset["TK_iwct"] = HIRS._create_temperature_array_3d(height, "Temperature internal warm calibration target (IWCT)", "temp_iwt", ["prt_number_iwt", "y", "prt_reading"])
        dataset["TK_patch_analog"] = HIRS._create_temperature_vector(height, "temperature_patch_analog_K", "temp_an_pch", np.NaN)

        default_array = DefaultData.create_default_array(PRT_READING, height, np.float32, dims_names=["y", "prt_reading"], fill_value=np.NaN)
        variable = Variable(["y", "prt_reading"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = "Temperature patch (expanded)"
        variable.attrs["orig_name"] = "temp_patch_exp"
        tu.add_units(variable, "K")
        dataset["TK_patch_exp"] = variable

        dataset["TK_patch_full"] = HIRS._create_temperature_vector(height, "temperature_patch_full_range_K", "temp_patch_full", np.NaN)
        dataset["TK_tlscp_prim"] = HIRS._create_temperature_vector(height, "temperature_telescope_primary_K", "temp_primtlscp", np.NaN)
        dataset["TK_tlscp_sec"] = HIRS._create_temperature_vector(height, "temperature_telescope_secondary_K", "temp_sectlscp", np.NaN)
        dataset["TK_tlscp_tert"] = HIRS._create_temperature_vector(height, "temperature_telescope_tertiary_K")
        dataset["TK_scanmirror"] = HIRS._create_temperature_vector(height, "temperature_scanmirror_K", "temp_scanmirror", np.NaN)
        dataset["TK_scanmirror_analog"] = HIRS._create_temperature_vector(height, "temperature_scanmirror_analog_K", "temp_an_scnm", np.NaN)
        dataset["TK_scanmotor"] = HIRS._create_temperature_vector(height, "temperature_scanmotor_K", "temp_scanmotor", np.NaN)

        dataset["u_TK_baseplate"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_baseplate_K")
        dataset["u_TK_ch"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_coolerhousing_K")
        dataset["u_TK_elec"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_electronics_K")
        dataset["u_TK_fsr"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_first_stage_radiator_K")
        dataset["u_TK_fwh"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_filter_wheel_housing_K")
        dataset["u_TK_fwm"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_filter_wheel_monitor_K")
        dataset["u_TK_icct"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_internal_cold_calibration_target_K")
        dataset["u_TK_iwct"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_internal_warm_calibration_target_K")
        dataset["u_TK_patch_exp"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_patch_expanded_scale_K")
        dataset["u_TK_patch_full"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_patch_full_range_K")
        dataset["u_TK_tlscp_prim"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_telescope_primary_K")
        dataset["u_TK_tlscp_sec"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_telescope_secondary_K")
        dataset["u_TK_tlscp_tert"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_telescope_tertiary_K")
        dataset["u_TK_scanmirror"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_scanmirror_K")
        dataset["u_TK_scanmotor"] = HIRS._create_temperature_vector(height, "uncertainty_temperature_scanmotor_K")

        dataset["u_solar_zenith_angle"] = HIRS._create_geo_angle_uncertainty_variable("uncertainty_solar_zenith_angle", height, FILL_VALUE)
        dataset["u_solar_azimuth_angle"] = HIRS._create_geo_angle_uncertainty_variable("uncertainty_solar_azimuth_angle", height, FILL_VALUE)
        dataset["u_satellite_zenith_angle"] = HIRS._create_geo_angle_uncertainty_variable("uncertainty_satellite_zenith_angle", height, FILL_VALUE)
        dataset["u_satellite_azimuth_angle"] = HIRS._create_geo_angle_uncertainty_variable("uncertainty_local_azimuth_angle", height, FILL_VALUE)

        # u_c_earth_chan_corr
        dataset["u_c_earth_chan_corr"] = HIRS._create_channel_correlation_variable("u_c_earth channel correlations")

        # u_c_space
        default_array = DefaultData.create_default_array(NUM_CALIBRATION_CYCLE, NUM_CHANNELS, np.uint16, dims_names=["channel", "calibration_cycle"])
        variable = Variable(["channel", "calibration_cycle"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        tu.add_units(variable, "count")
        tu.add_scale_factor(variable, 0.005)
        variable.attrs["long_name"] = "uncertainty counts for space views"
        variable.attrs["ancilliary_variables"] = "u_c_space_chan_corr"
        variable.attrs["channels_affected"] = "all"
        variable.attrs["parameter"] = "C_s"
        variable.attrs["pdf_shape"] = "gaussian"
        dataset["u_c_space"] = variable

        # u_c_space_chan_corr
        dataset["u_c_space_chan_corr"] = HIRS._create_channel_correlation_variable("u_c_space channel correlations")

        # u_Earthshine
        dataset["u_Earthshine"] = tu.create_scalar_float_variable()
        dataset["u_O_Re"] = tu.create_scalar_float_variable()
        dataset["u_O_TIWCT"] = tu.create_scalar_float_variable()

        # u_O_TPRT
        variable = Variable([], np.uint16(65535))
        tu.add_fill_value(variable, 65535)
        tu.add_scale_factor(variable, 0.01)
        tu.add_units(variable, "K")
        variable.attrs["channels_affected"] = "all"
        variable.attrs[corr.SCAN_CORR_FORM] = corr.RECT
        variable.attrs[corr.SCAN_CORR_UNIT] = corr.PIXEL
        variable.attrs[corr.SCAN_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.TIME_CORR_FORM] = corr.RECT
        variable.attrs[corr.TIME_CORR_UNIT] = corr.LINE
        variable.attrs[corr.TIME_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs[corr.IMG_CORR_FORM] = corr.RECT
        variable.attrs[corr.IMG_CORR_UNIT] = corr.IMG
        variable.attrs[corr.IMG_CORR_SCALE] = [-np.inf, np.inf]
        variable.attrs["parameter"] = "O_TPRT"
        variable.attrs["pdf_shape"] = "gaussian"
        variable.attrs["short_name"] = "O_TPRT"
        variable.attrs["ancilliary_variables"] = "u_O_TPRT_chan_corr"
        dataset["u_O_TPRT"] = variable

        # u_O_TPRT_chan_corr
        dataset["u_O_TPRT_chan_corr"] = HIRS._create_channel_correlation_variable("u_O_TPRT channel correlations")

        dataset["u_Rself"] = tu.create_scalar_float_variable()
        dataset["u_Rselfparams"] = tu.create_scalar_float_variable()
        dataset["u_SRF_calib"] = tu.create_scalar_float_variable()
        dataset["u_d_PRT"] = tu.create_scalar_float_variable()
        dataset["u_electronics"] = tu.create_scalar_float_variable()
        dataset["u_extraneous_periodic"] = tu.create_scalar_float_variable()
        dataset["u_nonlinearity"] = tu.create_scalar_float_variable()
        dataset["emissivity"] = tu.create_scalar_float_variable("emissivity", units="1")
        dataset["temp_corr_slope"] = tu.create_scalar_float_variable("Slope for effective temperature correction", units="1")
        dataset["temp_corr_offset"] = tu.create_scalar_float_variable("Offset for effective temperature correction", units="1")

    @staticmethod
    def _add_HIRS2_flag_variables(dataset, height):
        pass

    @staticmethod
    def _add_HIRS3_flag_variables(dataset, height):
        pass

    @staticmethod
    def _add_HIRS4_flag_variables(dataset, height):
        pass

    @staticmethod
    def _create_temperature_array_3d(height, long_name, orig_name, dim_names):
        default_array = DefaultData.create_default_array_3d(PRT_READING, height, PRT_NUMBER, np.float32, fill_value=np.NaN, dims_names=dim_names)
        variable = Variable(["prt_number", "y", "prt_reading"], default_array)
        tu.add_fill_value(variable, np.NaN)
        tu.add_units(variable, "K")
        variable.attrs["long_name"] = long_name
        variable.attrs["orig_name"] = orig_name
        return variable

    @staticmethod
    def _create_channel_correlation_variable(long_name):
        data_array = DefaultData.create_default_array(NUM_CHANNELS, NUM_CHANNELS, np.float32, fill_value=np.NaN)
        variable = Variable(["channel", "channel"], data_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["long_name"] = long_name
        return variable

    @staticmethod
    def _create_temperature_vector(height, long_name, orig_name=None, fill_value=None):
        variable = HIRS._create_float32_vector(fill_value, height, long_name, orig_name)
        tu.add_units(variable, "K")
        return variable

    @staticmethod
    def _create_float32_vector(fill_value, height, long_name, orig_name):
        default_array = DefaultData.create_default_vector(height, np.float32, fill_value=fill_value)
        variable = Variable(["y"], default_array)
        if fill_value is None:
            tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        else:
            tu.add_fill_value(variable, fill_value)
        variable.attrs["long_name"] = long_name
        if orig_name is not None:
            variable.attrs["orig_name"] = orig_name
        return variable

    @staticmethod
    def _create_float32_angle_vector(fill_value, height, long_name, orig_name):
        variable = HIRS._create_float32_vector(fill_value, height, long_name, orig_name)
        tu.add_units(variable, "degree")
        return variable

    @staticmethod
    def _create_counts_vector(height, standard_name):
        variable = HIRS._create_int32_vector(height, standard_name)
        tu.add_units(variable, "count")
        return variable

    @staticmethod
    def _create_int32_vector(height, standard_name=None, long_name=None, orig_name=None):
        default_array = DefaultData.create_default_vector(height, np.int32)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int32))
        if standard_name is not None:
            variable.attrs["standard_name"] = standard_name
        if long_name is not None:
            variable.attrs["long_name"] = long_name
        if orig_name is not None:
            variable.attrs["orig_name"] = orig_name

        return variable

    @staticmethod
    def _create_counts_uncertainty_vector(height, standard_name):
        default_array = DefaultData.create_default_vector(height, np.float32)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = standard_name
        tu.add_units(variable, "count")
        return variable

    @staticmethod
    def _create_3d_rad_uncertainty_variable(height, standard_name):
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_RAD_CHANNELS, np.float32, dims_names=["rad_channel", "y", "x"])
        variable = Variable(["rad_channel", "y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = standard_name
        return variable

    @staticmethod
    def _create_3d_bt_uncertainty_variable(height, standard_name):
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_CHANNELS, np.float32, dims_names=["channel", "y", "x"])
        variable = Variable(["channel", "y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = standard_name
        return variable

    @staticmethod
    def _create_angle_variable(height, standard_name):
        variable = tu.create_float_variable(SWATH_WIDTH, height, standard_name)
        tu.add_units(variable, "degree")
        return variable

    @staticmethod
    def _create_geo_angle_variable(standard_name, height, orig_name=None):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = standard_name
        if orig_name is not None:
            variable.attrs["orig_name"] = orig_name

        tu.add_units(variable, "degree")
        tu.add_encoding(variable, np.uint16, DefaultData.get_default_fill_value(np.uint16), 0.01, -180.0)
        return variable

    @staticmethod
    def _create_geo_angle_uncertainty_variable(standard_name, height, fill_value, orig_name=None):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=fill_value)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, fill_value)
        variable.attrs["standard_name"] = standard_name
        if orig_name is not None:
            variable.attrs["orig_name"] = orig_name

        tu.add_units(variable, "degree")
        return variable
