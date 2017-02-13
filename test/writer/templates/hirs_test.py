import unittest

import numpy as np
import xarray as xr

from writer.default_data import DefaultData
from writer.templates.hirs import HIRS


class HIRSTest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        HIRS.add_original_variables(ds, 6)

        latitude = ds.variables["latitude"]
        self.assertEqual((6, 56), latitude.shape)
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual(-32768.0, latitude.data[0, 0])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])

        longitude = ds.variables["longitude"]
        self.assertEqual((6, 56), latitude.shape)
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual(-32768.0, longitude.data[0, 0])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])

        bt = ds.variables["bt"]
        self.assertEqual((19, 6, 56), bt.shape)
        self.assertEqual(-999.0, bt.data[0, 2, 1])
        self.assertEqual(-999.0, bt.attrs["_FillValue"])
        self.assertEqual("toa_brightness_temperature", bt.attrs["standard_name"])
        self.assertEqual("K", bt.attrs["units"])
        self.assertEqual("scnlinf qualind linqualflags chqualflags mnfrqualflags", bt.attrs["ancilliary_variables"])

        c_earth = ds.variables["c_earth"]
        self.assertEqual((20, 6, 56), c_earth.shape)
        self.assertEqual(99999, c_earth.data[0, 2, 3])
        self.assertEqual(99999, c_earth.attrs["_FillValue"])
        self.assertEqual("counts_earth", c_earth.attrs["long_name"])
        self.assertEqual("true", c_earth.attrs["_Unsigned"])
        self.assertEqual("count", c_earth.attrs["units"])
        self.assertEqual("scnlinf qualind linqualflags chqualflags mnfrqualflags", c_earth.attrs["ancilliary_variables"])

        l_earth = ds.variables["L_earth"]
        self.assertEqual((20, 6, 56), l_earth.shape)
        self.assertEqual(-999.0, l_earth.data[0, 2, 4])
        self.assertEqual(-999.0, l_earth.attrs["_FillValue"])
        self.assertEqual("toa_outgoing_inband_radiance", l_earth.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", l_earth.attrs["units"])
        self.assertEqual("scnlinf qualind linqualflags chqualflags mnfrqualflags", l_earth.attrs["ancilliary_variables"])

        sat_za = ds.variables["sat_za"]
        self.assertEqual((6, 56), sat_za.shape)
        self.assertEqual(-999.0, sat_za.data[2, 2])
        self.assertEqual(-999.0, sat_za.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_za.attrs["standard_name"])
        self.assertEqual("degree", sat_za.attrs["units"])

        sat_aa = ds.variables["sat_aa"]
        self.assertEqual((6, 56), sat_aa.shape)
        self.assertEqual(-999.0, sat_aa.data[5, 5])
        self.assertEqual(-999.0, sat_aa.attrs["_FillValue"])
        self.assertEqual("sensor_azimuth_angle", sat_aa.attrs["standard_name"])
        self.assertEqual("local_azimuth_angle", sat_aa.attrs["long_name"])
        self.assertEqual("degree", sat_aa.attrs["units"])

        sol_za = ds.variables["sol_za"]
        self.assertEqual((6, 56), sol_za.shape)
        self.assertEqual(-999.0, sol_za.data[3, 3])
        self.assertEqual(-999.0, sol_za.attrs["_FillValue"])
        self.assertEqual("solar_zenith_angle", sol_za.attrs["standard_name"])
        self.assertEqual("degree", sol_za.attrs["units"])

        sol_aa = ds.variables["sol_aa"]
        self.assertEqual((6, 56), sol_aa.shape)
        self.assertEqual(-999.0, sol_aa.data[4, 4])
        self.assertEqual(-999.0, sol_aa.attrs["_FillValue"])
        self.assertEqual("solar_azimuth_angle", sol_aa.attrs["standard_name"])
        self.assertEqual("degree", sol_aa.attrs["units"])

        scanline = ds.variables["scanline"]
        self.assertEqual((6,), scanline.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.data[3])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.attrs["_FillValue"])
        self.assertEqual("scanline_number", scanline.attrs["long_name"])
        self.assertEqual("count", scanline.attrs["units"])

        time = ds.variables["time"]
        self.assertEqual((6,), time.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), time.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), time.attrs["_FillValue"])
        self.assertEqual("time", time.attrs["standard_name"])
        self.assertEqual("Acquisition time in seconds since 1970-01-01 00:00:00", time.attrs["long_name"])
        self.assertEqual("true", time.attrs["_Unsigned"])
        self.assertEqual("s", time.attrs["units"])

        scnlinf = ds.variables["scnlinf"]
        self.assertEqual((6,), scnlinf.shape)
        self.assertEqual(9, scnlinf.data[4])
        self.assertEqual(9, scnlinf.attrs["_FillValue"])
        self.assertEqual("0, 1, 2, 3", scnlinf.attrs["flag_values"])
        self.assertEqual("earth_view space_view icct_view iwct_view", scnlinf.attrs["flag_meanings"])
        self.assertEqual("status_flag", scnlinf.attrs["standard_name"])
        self.assertEqual("scanline_bitfield", scnlinf.attrs["long_name"])

        qualind = ds.variables["qualind"]
        self.assertEqual((6,), qualind.shape)
        self.assertEqual(0, qualind.data[5])
        self.assertEqual("1, 2, 4, 8, 16, 32, 64, 128", qualind.attrs["flag_masks"])
        self.assertEqual("do_not_use_scan time_sequence_error data_gap_preceding_scan no_calibration no_earth_location clock_update status_changed line_incomplete", qualind.attrs["flag_meanings"])
        self.assertEqual("status_flag", qualind.attrs["standard_name"])
        self.assertEqual("quality_indicator_bitfield", qualind.attrs["long_name"])

        linqualflags = ds.variables["linqualflags"]
        self.assertEqual((6,), linqualflags.shape)
        self.assertEqual(0, linqualflags.data[0])
        self.assertEqual("256, 512, 1024, 2048, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456", linqualflags.attrs["flag_masks"])
        self.assertEqual("time_field_bad time_field_bad_not_inf inconsistent_sequence scan_time_repeat uncalib_bad_time calib_few_scans uncalib_bad_prt calib_marginal_prt uncalib_channels uncalib_inst_mode quest_ant_black_body zero_loc bad_loc_time bad_loc_marginal bad_loc_reason bad_loc_ant", linqualflags.attrs["flag_meanings"])
        self.assertEqual("status_flag", linqualflags.attrs["standard_name"])
        self.assertEqual("scanline_quality_flags_bitfield", linqualflags.attrs["long_name"])

        chqualflags = ds.variables["chqualflags"]
        self.assertEqual((6, 19), chqualflags.shape)
        self.assertEqual(0, chqualflags.data[1, 2])
        self.assertEqual("status_flag", chqualflags.attrs["standard_name"])
        self.assertEqual("channel_quality_flags_bitfield", chqualflags.attrs["long_name"])

        mnfrqualflags = ds.variables["mnfrqualflags"]
        self.assertEqual((6, 23), mnfrqualflags.shape)
        self.assertEqual(0, mnfrqualflags.data[2, 5])
        self.assertEqual("status_flag", mnfrqualflags.attrs["standard_name"])
        self.assertEqual("minor_frame_quality_flags_bitfield", mnfrqualflags.attrs["long_name"])

    def test_get_swath_width(self):
        self.assertEqual(56, HIRS.get_swath_width())

    def test_add_uncertainty_variables(self):
        ds = xr.Dataset()
        HIRS.add_full_fcdr_variables(ds, 7)

        u_lat = ds.variables["u_lat"]
        self.assertEqual((7, 56), u_lat.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_lat.data[3, 3])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_lat.attrs["_FillValue"])
        self.assertEqual("uncertainty_latitude", u_lat.attrs["standard_name"])
        self.assertEqual("degree", u_lat.attrs["units"])

        u_lon = ds.variables["u_lon"]
        self.assertEqual((7, 56), u_lon.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_lon.data[4, 4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_lon.attrs["_FillValue"])
        self.assertEqual("uncertainty_longitude", u_lon.attrs["standard_name"])
        self.assertEqual("degree", u_lon.attrs["units"])

        u_time = ds.variables["u_time"]
        self.assertEqual((7, 56), u_time.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_time.data[5, 5])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_time.attrs["_FillValue"])
        self.assertEqual("uncertainty_time", u_time.attrs["standard_name"])
        self.assertEqual("s", u_time.attrs["units"])

        u_c_earth = ds.variables["u_c_earth"]
        self.assertEqual((20, 7, 56), u_c_earth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_c_earth.data[6, 6, 6])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_c_earth.attrs["_FillValue"])
        self.assertEqual("uncertainty_counts_Earth", u_c_earth.attrs["standard_name"])
        self.assertEqual("count", u_c_earth.attrs["units"])

        u_L_earth_random = ds.variables["u_L_earth_random"]
        self.assertEqual((20, 7, 56), u_L_earth_random.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_random.data[7, 0, 7])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_random.attrs["_FillValue"])
        self.assertEqual("uncertainty_radiance_Earth_random", u_L_earth_random.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", u_L_earth_random.attrs["units"])

        u_L_earth_sr = ds.variables["u_L_earth_structuredrandom"]
        self.assertEqual((20, 7, 56), u_L_earth_sr.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_sr.data[8, 1, 8])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_sr.attrs["_FillValue"])
        self.assertEqual("uncertainty_radiance_Earth_structured_random", u_L_earth_sr.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", u_L_earth_sr.attrs["units"])

        u_L_earth_sys = ds.variables["u_L_earth_systematic"]
        self.assertEqual((20, 7, 56), u_L_earth_sys.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_sys.data[9, 2, 9])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_sys.attrs["_FillValue"])
        self.assertEqual("uncertainty_radiance_Earth_systematic", u_L_earth_sys.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", u_L_earth_sys.attrs["units"])

        u_L_earth_total = ds.variables["u_L_earth_total"]
        self.assertEqual((20, 7, 56), u_L_earth_total.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_total.data[10, 3, 10])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_total.attrs["_FillValue"])
        self.assertEqual("uncertainty_radiance_Earth_total", u_L_earth_total.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", u_L_earth_total.attrs["units"])

        S_u_L_earth = ds.variables["S_u_L_earth"]
        self.assertEqual((20, 20), S_u_L_earth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), S_u_L_earth.data[11, 4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), S_u_L_earth.attrs["_FillValue"])
        self.assertEqual("covariance_radiance_Earth", S_u_L_earth.attrs["standard_name"])

        u_bt_random = ds.variables["u_bt_random"]
        self.assertEqual((19, 7, 56), u_bt_random.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_random.data[13, 6, 13])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_random.attrs["_FillValue"])
        self.assertEqual("uncertainty_bt_random", u_bt_random.attrs["standard_name"])
        self.assertEqual("K", u_bt_random.attrs["units"])

        u_bt_structuredrandom = ds.variables["u_bt_structuredrandom"]
        self.assertEqual((19, 7, 56), u_bt_structuredrandom.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_structuredrandom.data[14, 0, 14])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_structuredrandom.attrs["_FillValue"])
        self.assertEqual("uncertainty_bt_structured_random", u_bt_structuredrandom.attrs["standard_name"])
        self.assertEqual("K", u_bt_structuredrandom.attrs["units"])

        u_bt_sys = ds.variables["u_bt_systematic"]
        self.assertEqual((19, 7, 56), u_bt_sys.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_sys.data[15, 1, 15])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_sys.attrs["_FillValue"])
        self.assertEqual("uncertainty_bt_systematic", u_bt_sys.attrs["standard_name"])
        self.assertEqual("K", u_bt_sys.attrs["units"])

        u_bt_total = ds.variables["u_bt_total"]
        self.assertEqual((19, 7, 56), u_bt_total.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_total.data[15, 1, 15])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_total.attrs["_FillValue"])
        self.assertEqual("uncertainty_bt_total", u_bt_total.attrs["standard_name"])
        self.assertEqual("K", u_bt_total.attrs["units"])

        S_bt = ds.variables["S_bt"]
        self.assertEqual((19,19), S_bt.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), S_bt.data[12, 3])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), S_bt.attrs["_FillValue"])
        self.assertEqual("covariance_brightness_temperature", S_bt.attrs["standard_name"])

        calcof = ds.variables["calcof"]
        self.assertEqual((3, 7, 56), calcof.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), calcof.data[0, 2, 16])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), calcof.attrs["_FillValue"])
        self.assertEqual("calibration_coefficients", calcof.attrs["standard_name"])

        u_calcof = ds.variables["u_calcof"]
        self.assertEqual((3, 7, 56), u_calcof.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_calcof.data[1, 3, 17])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_calcof.attrs["_FillValue"])
        self.assertEqual("uncertainty_calibration_coefficients", u_calcof.attrs["standard_name"])

        self._assert_line_counts_variable(ds, "Tc_baseplate", "temperature_baseplate_counts")
        self._assert_line_counts_variable(ds, "Tc_ch", "temperature_coolerhousing_counts")
        self._assert_line_counts_variable(ds, "Tc_elec", "temperature_electronics_counts")
        self._assert_line_counts_variable(ds, "Tc_fsr", "temperature_first_stage_radiator_counts")
        self._assert_line_counts_variable(ds, "Tc_fwh", "temperature_filter_wheel_housing_counts")
        self._assert_line_counts_variable(ds, "Tc_fwm", "temperature_filter_wheel_monitor_counts")
        self._assert_line_counts_variable(ds, "Tc_icct", "temperature_internal_cold_calibration_target_counts")
        self._assert_line_counts_variable(ds, "Tc_iwct", "temperature_internal_warm_calibration_target_counts")
        self._assert_line_counts_variable(ds, "Tc_patch_exp", "temperature_patch_expanded_scale_counts")
        self._assert_line_counts_variable(ds, "Tc_patch_full", "temperature_patch_full_range_counts")
        self._assert_line_counts_variable(ds, "Tc_tlscp_prim", "temperature_telescope_primary_counts")
        self._assert_line_counts_variable(ds, "Tc_tlscp_sec", "temperature_telescope_secondary_counts")
        self._assert_line_counts_variable(ds, "Tc_tlscp_tert", "temperature_telescope_tertiary_counts")
        self._assert_line_counts_variable(ds, "Tc_scanmirror", "temperature_scanmirror_counts")
        self._assert_line_counts_variable(ds, "Tc_scanmotor", "temperature_scanmotor_counts")

        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_baseplate", "uncertainty_temperature_baseplate_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_ch", "uncertainty_temperature_coolerhousing_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_elec", "uncertainty_temperature_electronics_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_fsr", "uncertainty_temperature_first_stage_radiator_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_fwh", "uncertainty_temperature_filter_wheel_housing_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_fwm", "uncertainty_temperature_filter_wheel_monitor_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_icct", "uncertainty_temperature_internal_cold_calibration_target_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_iwct", "uncertainty_temperature_internal_warm_calibration_target_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_patch_exp", "uncertainty_temperature_patch_expanded_scale_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_patch_full", "uncertainty_temperature_patch_full_range_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_tlscp_prim", "uncertainty_temperature_telescope_primary_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_tlscp_sec", "uncertainty_temperature_telescope_secondary_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_tlscp_tert", "uncertainty_temperature_telescope_tertiary_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_scanmirror", "uncertainty_temperature_scanmirror_counts")
        self._assert_line_counts_uncertainty_variable(ds, "u_Tc_scanmotor", "uncertainty_temperature_scanmotor_counts")

        self._assert_line_temperature_variable(ds, "TK_baseplate", "temperature_baseplate_K")
        self._assert_line_temperature_variable(ds, "TK_ch", "temperature_coolerhousing_K")
        self._assert_line_temperature_variable(ds, "TK_elec", "temperature_electronics_K")
        self._assert_line_temperature_variable(ds, "TK_fsr", "temperature_first_stage_radiator_K")
        self._assert_line_temperature_variable(ds, "TK_fwh", "temperature_filter_wheel_housing_K")
        self._assert_line_temperature_variable(ds, "TK_fwm", "temperature_filter_wheel_monitor_K")
        self._assert_line_temperature_variable(ds, "TK_icct", "temperature_internal_cold_calibration_target_K")
        self._assert_line_temperature_variable(ds, "TK_iwct", "temperature_internal_warm_calibration_target_K")
        self._assert_line_temperature_variable(ds, "TK_patch_exp", "temperature_patch_expanded_scale_K")
        self._assert_line_temperature_variable(ds, "TK_patch_full", "temperature_patch_full_range_K")
        self._assert_line_temperature_variable(ds, "TK_tlscp_prim", "temperature_telescope_primary_K")
        self._assert_line_temperature_variable(ds, "TK_tlscp_sec", "temperature_telescope_secondary_K")
        self._assert_line_temperature_variable(ds, "TK_tlscp_tert", "temperature_telescope_tertiary_K")
        self._assert_line_temperature_variable(ds, "TK_scanmirror", "temperature_scanmirror_K")
        self._assert_line_temperature_variable(ds, "TK_scanmotor", "temperature_scanmotor_K")

        self._assert_line_temperature_variable(ds, "u_TK_baseplate", "uncertainty_temperature_baseplate_K")
        self._assert_line_temperature_variable(ds, "u_TK_ch", "uncertainty_temperature_coolerhousing_K")
        self._assert_line_temperature_variable(ds, "u_TK_elec", "uncertainty_temperature_electronics_K")
        self._assert_line_temperature_variable(ds, "u_TK_fsr", "uncertainty_temperature_first_stage_radiator_K")
        self._assert_line_temperature_variable(ds, "u_TK_fwh", "uncertainty_temperature_filter_wheel_housing_K")
        self._assert_line_temperature_variable(ds, "u_TK_fwm", "uncertainty_temperature_filter_wheel_monitor_K")
        self._assert_line_temperature_variable(ds, "u_TK_icct", "uncertainty_temperature_internal_cold_calibration_target_K")
        self._assert_line_temperature_variable(ds, "u_TK_iwct", "uncertainty_temperature_internal_warm_calibration_target_K")
        self._assert_line_temperature_variable(ds, "u_TK_patch_exp", "uncertainty_temperature_patch_expanded_scale_K")
        self._assert_line_temperature_variable(ds, "u_TK_patch_full", "uncertainty_temperature_patch_full_range_K")
        self._assert_line_temperature_variable(ds, "u_TK_tlscp_prim", "uncertainty_temperature_telescope_primary_K")
        self._assert_line_temperature_variable(ds, "u_TK_tlscp_sec", "uncertainty_temperature_telescope_secondary_K")
        self._assert_line_temperature_variable(ds, "u_TK_tlscp_tert", "uncertainty_temperature_telescope_tertiary_K")
        self._assert_line_temperature_variable(ds, "u_TK_scanmirror", "uncertainty_temperature_scanmirror_K")
        self._assert_line_temperature_variable(ds, "u_TK_scanmotor", "uncertainty_temperature_scanmotor_K")

        u_sol_za = ds.variables["u_sol_za"]
        self.assertEqual((7, 56), u_sol_za.shape)
        self.assertEqual(-999.0, u_sol_za.data[4, 4])
        self.assertEqual(-999.0, u_sol_za.attrs["_FillValue"])
        self.assertEqual("uncertainty_solar_zenith_angle", u_sol_za.attrs["standard_name"])
        self.assertEqual("degree", u_sol_za.attrs["units"])

        u_sol_aa = ds.variables["u_sol_aa"]
        self.assertEqual((7, 56), u_sol_aa.shape)
        self.assertEqual(-999.0, u_sol_aa.data[5, 5])
        self.assertEqual(-999.0, u_sol_aa.attrs["_FillValue"])
        self.assertEqual("uncertainty_solar_azimuth_angle", u_sol_aa.attrs["standard_name"])
        self.assertEqual("degree", u_sol_aa.attrs["units"])

        u_sat_za = ds.variables["u_sat_za"]
        self.assertEqual((7, 56), u_sat_za.shape)
        self.assertEqual(-999.0, u_sat_za.data[5, 5])
        self.assertEqual(-999.0, u_sat_za.attrs["_FillValue"])
        self.assertEqual("uncertainty_satellite_zenith_angle", u_sat_za.attrs["standard_name"])
        self.assertEqual("degree", u_sat_za.attrs["units"])

        u_sat_aa = ds.variables["u_sat_aa"]
        self.assertEqual((7, 56), u_sat_aa.shape)
        self.assertEqual(-999.0, u_sat_aa.data[6, 6])
        self.assertEqual(-999.0, u_sat_aa.attrs["_FillValue"])
        self.assertEqual("uncertainty_local_azimuth_angle", u_sat_aa.attrs["standard_name"])
        self.assertEqual("degree", u_sat_aa.attrs["units"])

    def _assert_line_counts_variable(self, ds, name, standard_name):
        variable = ds.variables[name]
        self.assertEqual((7,), variable.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), variable.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), variable.attrs["_FillValue"])
        self.assertEqual(standard_name, variable.attrs["standard_name"])
        self.assertEqual("count", variable.attrs["units"])

    def _assert_line_temperature_variable(self, ds, name, standard_name):
        variable = ds.variables[name]
        self.assertEqual((7,), variable.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), variable.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), variable.attrs["_FillValue"])
        self.assertEqual(standard_name, variable.attrs["standard_name"])
        self.assertEqual("K", variable.attrs["units"])

    def _assert_line_counts_uncertainty_variable(self, ds, name, standard_name):
        variable = ds.variables[name]
        self.assertEqual((7,), variable.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), variable.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), variable.attrs["_FillValue"])
        self.assertEqual(standard_name, variable.attrs["standard_name"])
        self.assertEqual("count", variable.attrs["units"])

