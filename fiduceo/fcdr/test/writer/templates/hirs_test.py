import unittest

import numpy as np
import xarray as xr

from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.hirs import HIRS


class HIRSTest(unittest.TestCase):
    def test_add_full_fcdr_variables(self):
        ds = xr.Dataset()
        HIRS.add_full_fcdr_variables(ds, 7)

        c_earth = ds.variables["c_earth"]
        self.assertEqual((20, 7, 56), c_earth.shape)
        self.assertEqual(65535, c_earth.data[0, 2, 3])
        self.assertEqual(65535, c_earth.attrs["_FillValue"])
        self.assertEqual("counts_earth", c_earth.attrs["long_name"])
        self.assertEqual("count", c_earth.attrs["units"])
        self.assertEqual("scnlinf qualind linqualflags chqualflags mnfrqualflags", c_earth.attrs["ancilliary_variables"])

        l_earth = ds.variables["L_earth"]
        self.assertEqual((20, 7, 56), l_earth.shape)
        self.assertTrue(np.isnan(l_earth.data[0, 2, 4]))
        self.assertTrue(np.isnan(l_earth.attrs["_FillValue"]))
        self.assertEqual("toa_outgoing_inband_radiance", l_earth.attrs["standard_name"])
        self.assertEqual("W/Hz/m ** 2/sr", l_earth.attrs["units"])
        self.assertEqual("Channel radiance, NOAA/EUMETSAT calibrated", l_earth.attrs["long_name"])
        self.assertEqual("radiance", l_earth.attrs["orig_name"])
        self.assertEqual("scnlinf qualind linqualflags chqualflags mnfrqualflags", l_earth.attrs["ancilliary_variables"])

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
        self.assertEqual((19, 337), u_c_earth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_c_earth.data[6, 6])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_c_earth.attrs["_FillValue"])
        self.assertEqual("uncertainty counts for Earth views", u_c_earth.attrs["long_name"])
        self.assertEqual("count", u_c_earth.attrs["units"])
        self.assertEqual("u_c_earth_chan_corr", u_c_earth.attrs["ancilliary_variables"])
        self.assertEqual("all", u_c_earth.attrs["channels_affected"])
        self.assertEqual("C_E", u_c_earth.attrs["parameter"])
        self.assertEqual("gaussian", u_c_earth.attrs["pdf_shape"])
        self.assertEqual(0.005, u_c_earth.attrs["scale_factor"])

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
        self.assertEqual((19, 19), S_bt.shape)
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

        self._assert_line_int32_variable(ds, "navigation_status", standard_name="status_flag", long_name="Navigation status bit field", orig_name="hrs_navstat")

        platform_altitude = self._assert_line_float_variable(ds, "platform_altitude", long_name="Platform altitude", orig_name="hrs_scalti", fill_value=np.NaN)
        self.assertEqual("km", platform_altitude.attrs["units"])

        self._assert_line_angle_variable(ds, "platform_pitch_angle", long_name="Platform pitch angle", orig_name="hrs_pitchang", fill_value=np.NaN)
        self._assert_line_angle_variable(ds, "platform_roll_angle", long_name="Platform roll angle", orig_name="hrs_rollang", fill_value=np.NaN)
        self._assert_line_angle_variable(ds, "platform_yaw_angle", long_name="Platform yaw angle", orig_name="hrs_yawang", fill_value=np.NaN)

        self._assert_line_int32_variable(ds, "quality_flags", standard_name="status_flag", long_name="Quality indicator bit field", orig_name="hrs_qualind")

        scan_angles = ds.variables["scan_angles"]
        self.assertEqual((7, 168), scan_angles.shape)
        self.assertTrue(np.isnan(scan_angles.data[4, 18]))
        self.assertTrue(np.isnan(scan_angles.attrs["_FillValue"]))
        self.assertEqual("Scan angles", scan_angles.attrs["long_name"])
        self.assertEqual("hrs_ang", scan_angles.attrs["orig_name"])
        self.assertEqual("degree", scan_angles.attrs["units"])

        self._assert_line_int32_variable(ds, "scanline_number", long_name="scanline number", orig_name="hrs_scnlin")
        self._assert_line_int32_variable(ds, "scanline_position", long_name="Scanline position number in 32 second cycle", orig_name="hrs_scnpos")

        sec_o_cal_coeff = ds.variables["second_original_calibration_coefficients"]
        self.assertEqual((7, 60), sec_o_cal_coeff.shape)
        self.assertTrue(np.isnan(sec_o_cal_coeff.data[4, 18]))
        self.assertTrue(np.isnan(sec_o_cal_coeff.attrs["_FillValue"]))
        self.assertEqual("Second original calibration coefficients (unsorted)", sec_o_cal_coeff.attrs["long_name"])
        self.assertEqual("hrs_scalcof", sec_o_cal_coeff.attrs["orig_name"])

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

        self._assert_line_temperature_variable(ds, "TK_baseplate", "Temperature baseplate", orig_name="temp_baseplate", fill_value=np.NaN)
        self._assert_line_temperature_variable(ds, "TK_baseplate_analog", "Temperature baseplate (analog)", "temp_an_baseplate", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_ch", "Temperature cooler housing", "temp_ch", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_elec", "Temperature electronics", "temp_elec", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_elec_analog", "Temperature electronics (analog)", "temp_an_el", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_radiator_analog", "temperature_radiator_analog_K", "temp_an_rd", np.NaN)

        self._assert_2d_temperature_variable(ds, "TK_fsr", "Temperature first stage radiator", "temp_fsr")

        self._assert_line_temperature_variable(ds, "TK_fwm", "Temperature filter wheel motor", "temp_fwm", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_fwm_analog", "Temperature filter wheel motor (analogue)", "temp_an_fwm", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_icct", "temperature_internal_cold_calibration_target_K")

        self._assert_3d_temperature_variable(ds, "TK_fwh", "Temperature filter wheel housing", "temp_fwh")
        self._assert_3d_temperature_variable(ds, "TK_iwct", "Temperature internal warm calibration target (IWCT)", "temp_iwt")
        self._assert_line_temperature_variable(ds, "TK_patch_analog", "temperature_patch_analog_K", "temp_an_pch", np.NaN)
        self._assert_2d_temperature_variable(ds, "TK_patch_exp", "Temperature patch (expanded)", "temp_patch_exp")
        self._assert_line_temperature_variable(ds, "TK_patch_full", "temperature_patch_full_range_K", "temp_patch_full", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_tlscp_prim", "temperature_telescope_primary_K", "temp_primtlscp", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_tlscp_sec", "temperature_telescope_secondary_K", "temp_sectlscp", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_tlscp_tert", "temperature_telescope_tertiary_K")
        self._assert_line_temperature_variable(ds, "TK_scanmirror", "temperature_scanmirror_K", "temp_scanmirror", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_scanmirror_analog", "temperature_scanmirror_analog_K", "temp_an_scnm", np.NaN)
        self._assert_line_temperature_variable(ds, "TK_scanmotor", "temperature_scanmotor_K", "temp_scanmotor", np.NaN)

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

        u_solar_zenith_angle = ds.variables["u_solar_zenith_angle"]
        self.assertEqual((7, 56), u_solar_zenith_angle.shape)
        self.assertEqual(-999.0, u_solar_zenith_angle.data[4, 4])
        self.assertEqual(-999.0, u_solar_zenith_angle.attrs["_FillValue"])
        self.assertEqual("uncertainty_solar_zenith_angle", u_solar_zenith_angle.attrs["standard_name"])
        self.assertEqual("degree", u_solar_zenith_angle.attrs["units"])

        u_solar_azimuth_angle = ds.variables["u_solar_azimuth_angle"]
        self.assertEqual((7, 56), u_solar_azimuth_angle.shape)
        self.assertEqual(-999.0, u_solar_azimuth_angle.data[5, 5])
        self.assertEqual(-999.0, u_solar_azimuth_angle.attrs["_FillValue"])
        self.assertEqual("uncertainty_solar_azimuth_angle", u_solar_azimuth_angle.attrs["standard_name"])
        self.assertEqual("degree", u_solar_azimuth_angle.attrs["units"])

        u_satellite_zenith_angle = ds.variables["u_satellite_zenith_angle"]
        self.assertEqual((7, 56), u_satellite_zenith_angle.shape)
        self.assertEqual(-999.0, u_satellite_zenith_angle.data[5, 5])
        self.assertEqual(-999.0, u_satellite_zenith_angle.attrs["_FillValue"])
        self.assertEqual("uncertainty_satellite_zenith_angle", u_satellite_zenith_angle.attrs["standard_name"])
        self.assertEqual("degree", u_satellite_zenith_angle.attrs["units"])

        u_satellite_azimuth_angle = ds.variables["u_satellite_azimuth_angle"]
        self.assertEqual((7, 56), u_satellite_azimuth_angle.shape)
        self.assertEqual(-999.0, u_satellite_azimuth_angle.data[6, 6])
        self.assertEqual(-999.0, u_satellite_azimuth_angle.attrs["_FillValue"])
        self.assertEqual("uncertainty_local_azimuth_angle", u_satellite_azimuth_angle.attrs["standard_name"])
        self.assertEqual("degree", u_satellite_azimuth_angle.attrs["units"])

        u_c_earth_chan_corr = ds.variables["u_c_earth_chan_corr"]
        self.assertEqual((19, 19), u_c_earth_chan_corr.shape)
        self.assertTrue(np.isnan(u_c_earth_chan_corr.data[11, 14]))
        self.assertTrue(np.isnan(u_c_earth_chan_corr.attrs["_FillValue"]))
        self.assertEqual("u_c_earth channel correlations", u_c_earth_chan_corr.attrs["long_name"])

        u_c_space = ds.variables["u_c_space"]
        self.assertEqual((19, 337), u_c_space.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_c_space.data[(12, 15)])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_c_space.attrs["_FillValue"])
        self.assertEqual("C_s", u_c_space.attrs["parameter"])
        self.assertEqual("gaussian", u_c_space.attrs["pdf_shape"])
        self.assertEqual(0.005, u_c_space.attrs["scale_factor"])
        self.assertEqual("count", u_c_space.attrs["units"])
        self.assertEqual("u_c_space_chan_corr", u_c_space.attrs["ancilliary_variables"])

        u_c_space_chan_corr = ds.variables["u_c_space_chan_corr"]
        self.assertEqual((19, 19), u_c_space_chan_corr.shape)
        self.assertTrue(np.isnan(u_c_space_chan_corr.data[11, 14]))
        self.assertTrue(np.isnan(u_c_space_chan_corr.attrs["_FillValue"]))
        self.assertEqual("u_c_space channel correlations", u_c_space_chan_corr.attrs["long_name"])

        u_earthshine = ds.variables["u_Earthshine"]
        self.assertEqual((), u_earthshine.shape)
        self.assertTrue(np.isnan(u_earthshine.data))
        self.assertTrue(np.isnan(u_earthshine.attrs["_FillValue"]))

        u_o_Re = ds.variables["u_O_Re"]
        self.assertEqual((), u_o_Re.shape)
        self.assertTrue(np.isnan(u_o_Re.data))
        self.assertTrue(np.isnan(u_o_Re.attrs["_FillValue"]))

        u_o_TIWCT = ds.variables["u_O_TIWCT"]
        self.assertEqual((), u_o_TIWCT.shape)
        self.assertTrue(np.isnan(u_o_TIWCT.data))
        self.assertTrue(np.isnan(u_o_TIWCT.attrs["_FillValue"]))

        u_o_TPRT = ds.variables["u_O_TPRT"]
        self.assertEqual((), u_o_TPRT.shape)
        self.assertEqual(65535, u_o_TPRT.data[()])
        self.assertEqual(65535, u_o_TPRT.attrs["_FillValue"])
        self.assertEqual("all", u_o_TPRT.attrs["channels_affected"])
        self.assertEqual("rectangle", u_o_TPRT.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_o_TPRT.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_o_TPRT.attrs["scan_correlation_scales"])
        self.assertEqual("rectangle", u_o_TPRT.attrs["time_correlation_form"])
        self.assertEqual("line", u_o_TPRT.attrs["time_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_o_TPRT.attrs["time_correlation_scales"])
        self.assertEqual("rectangle", u_o_TPRT.attrs["image_correlation_form"])
        self.assertEqual("images", u_o_TPRT.attrs["image_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_o_TPRT.attrs["image_correlation_scales"])
        self.assertEqual("O_TPRT", u_o_TPRT.attrs["parameter"])
        self.assertEqual("gaussian", u_o_TPRT.attrs["pdf_shape"])
        self.assertEqual(0.01, u_o_TPRT.attrs["scale_factor"])
        self.assertEqual("O_TPRT", u_o_TPRT.attrs["short_name"])
        self.assertEqual("K", u_o_TPRT.attrs["units"])
        self.assertEqual("u_O_TPRT_chan_corr", u_o_TPRT.attrs["ancilliary_variables"])

        u_o_TPRT_chan_corr = ds.variables["u_O_TPRT_chan_corr"]
        self.assertEqual((19, 19), u_o_TPRT_chan_corr.shape)
        self.assertTrue(np.isnan(u_o_TPRT_chan_corr.data[11, 14]))
        self.assertTrue(np.isnan(u_o_TPRT_chan_corr.attrs["_FillValue"]))
        self.assertEqual("u_O_TPRT channel correlations", u_o_TPRT_chan_corr.attrs["long_name"])

        u_Rself = ds.variables["u_Rself"]
        self.assertEqual((), u_Rself.shape)
        self.assertTrue(np.isnan(u_Rself.data))
        self.assertTrue(np.isnan(u_Rself.attrs["_FillValue"]))

        u_Rselfparams = ds.variables["u_Rselfparams"]
        self.assertEqual((), u_Rselfparams.shape)
        self.assertTrue(np.isnan(u_Rselfparams.data))
        self.assertTrue(np.isnan(u_Rselfparams.attrs["_FillValue"]))

        u_srf_calib = ds.variables["u_SRF_calib"]
        self.assertEqual((), u_srf_calib.shape)
        self.assertTrue(np.isnan(u_srf_calib.data))
        self.assertTrue(np.isnan(u_srf_calib.attrs["_FillValue"]))

        u_d_prt = ds.variables["u_d_PRT"]
        self.assertEqual((), u_d_prt.shape)
        self.assertTrue(np.isnan(u_d_prt.data))
        self.assertTrue(np.isnan(u_d_prt.attrs["_FillValue"]))

        u_electronics = ds.variables["u_electronics"]
        self.assertEqual((), u_electronics.shape)
        self.assertTrue(np.isnan(u_electronics.data))
        self.assertTrue(np.isnan(u_electronics.attrs["_FillValue"]))

        u_exp_periodic = ds.variables["u_extraneous_periodic"]
        self.assertEqual((), u_exp_periodic.shape)
        self.assertTrue(np.isnan(u_exp_periodic.data))
        self.assertTrue(np.isnan(u_exp_periodic.attrs["_FillValue"]))

        u_non_lin = ds.variables["u_nonlinearity"]
        self.assertEqual((), u_non_lin.shape)
        self.assertTrue(np.isnan(u_non_lin.data))
        self.assertTrue(np.isnan(u_non_lin.attrs["_FillValue"]))

        emissivity = ds.variables["emissivity"]
        self.assertEqual((), emissivity.shape)
        self.assertTrue(np.isnan(emissivity.data))
        self.assertTrue(np.isnan(emissivity.attrs["_FillValue"]))
        self.assertEqual("emissivity", emissivity.attrs["long_name"])
        self.assertEqual("1", emissivity.attrs["units"])

        temp_corr_slope = ds.variables["temp_corr_slope"]
        self.assertEqual((), temp_corr_slope.shape)
        self.assertTrue(np.isnan(temp_corr_slope.data))
        self.assertTrue(np.isnan(temp_corr_slope.attrs["_FillValue"]))
        self.assertEqual("Slope for effective temperature correction", temp_corr_slope.attrs["long_name"])
        self.assertEqual("1", temp_corr_slope.attrs["units"])

        temp_corr_offset = ds.variables["temp_corr_offset"]
        self.assertEqual((), temp_corr_offset.shape)
        self.assertTrue(np.isnan(temp_corr_offset.data))
        self.assertTrue(np.isnan(temp_corr_offset.attrs["_FillValue"]))
        self.assertEqual("Offset for effective temperature correction", temp_corr_offset.attrs["long_name"])
        self.assertEqual("1", temp_corr_offset.attrs["units"])

    def test_add_HIRS2_flag_variables(self):
        ds = xr.Dataset()
        HIRS._add_HIRS2_flag_variables(ds, 7)
        # @todo 2 tb/tb add assertions when Gerrit has defined the data for this sensor 2017-04-12

    def test_add_HIRS3_flag_variables(self):
        ds = xr.Dataset()
        HIRS._add_HIRS3_flag_variables(ds, 7)
        # @todo 2 tb/tb add assertions when Gerrit has defined the data for this sensor 2017-04-12

    def test_add_HIRS4_flag_variables(self):
        ds = xr.Dataset()
        HIRS._add_HIRS4_flag_variables(ds, 7)
        # @todo 2 tb/tb add assertions when Gerrit has defined the data for this sensor 2017-04-12

    def _assert_2d_temperature_variable(self, ds, name, long_name, orig_name):
        variable = ds.variables[name]
        self.assertEqual((7, 5), variable.shape)
        self.assertTrue(np.isnan(variable.data[6, 4]))
        self.assertTrue(np.isnan(variable.attrs["_FillValue"]))
        self.assertEqual(long_name, variable.attrs["long_name"])
        self.assertEqual(orig_name, variable.attrs["orig_name"])
        self.assertEqual("K", variable.attrs["units"])

    def _assert_3d_temperature_variable(self, ds, name, long_name, orig_name):
        variable = ds.variables[name]
        self.assertEqual((4, 7, 5), variable.shape)
        self.assertTrue(np.isnan(variable.data[2, 5, 3]))
        self.assertTrue(np.isnan(variable.attrs["_FillValue"]))
        self.assertEqual(long_name, variable.attrs["long_name"])
        self.assertEqual(orig_name, variable.attrs["orig_name"])
        self.assertEqual("K", variable.attrs["units"])

    def _assert_line_counts_variable(self, ds, name, standard_name):
        variable = self._assert_line_int32_variable(ds, name, standard_name)
        self.assertEqual("count", variable.attrs["units"])

    def _assert_line_int32_variable(self, ds, name, standard_name=None, long_name=None, orig_name=None):
        variable = ds.variables[name]
        self.assertEqual((7,), variable.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), variable.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), variable.attrs["_FillValue"])
        if standard_name is not None:
            self.assertEqual(standard_name, variable.attrs["standard_name"])

        if long_name is not None:
            self.assertEqual(long_name, variable.attrs["long_name"])

        if orig_name is not None:
            self.assertEqual(orig_name, variable.attrs["orig_name"])

        return variable

    def _assert_line_temperature_variable(self, ds, name, long_name, orig_name=None, fill_value=None):
        variable = self._assert_line_float_variable(ds, name, long_name=long_name, orig_name=orig_name, fill_value=fill_value)
        self.assertEqual("K", variable.attrs["units"])

    def _assert_line_counts_uncertainty_variable(self, ds, name, standard_name):
        variable = self._assert_line_float_variable(ds, name, standard_name=standard_name)
        self.assertEqual("count", variable.attrs["units"])

    def _assert_line_angle_variable(self, ds, name, long_name=None, orig_name=None, fill_value=None):
        variable = self._assert_line_float_variable(ds, name, long_name=long_name, orig_name=orig_name, fill_value=fill_value)
        self.assertEqual("degree", variable.attrs["units"])

    def _assert_line_float_variable(self, ds, name, standard_name=None, long_name=None, orig_name=None, fill_value=None):
        variable = ds.variables[name]
        self.assertEqual((7,), variable.shape)
        if fill_value is None:
            self.assertEqual(DefaultData.get_default_fill_value(np.float32), variable.data[4])
            self.assertEqual(DefaultData.get_default_fill_value(np.float32), variable.attrs["_FillValue"])
        elif np.isnan(fill_value):
            self.assertTrue(np.isnan(variable.data[4]))
            self.assertTrue(np.isnan(variable.attrs["_FillValue"]))
        else:
            self.assertEqual(fill_value, variable.data[4])
            self.assertEqual(fill_value, variable.attrs["_FillValue"])

        if standard_name is not None:
            self.assertEqual(standard_name, variable.attrs["standard_name"])

        if long_name is not None:
            self.assertEqual(long_name, variable.attrs["long_name"])

        if orig_name is not None:
            self.assertEqual(orig_name, variable.attrs["orig_name"])
        return variable
