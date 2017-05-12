import unittest

from fiduceo.fcdr.writer.fcdr_writer import FCDRWriter


class FCDRWriterTest(unittest.TestCase):
    def testCreateTemplateEasy_AMSUB(self):
        ds = FCDRWriter.createTemplateEasy('AMSUB', 2561)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(21, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self._verify_amsub_specific_variables(ds)

        # easy FCDR variables
        self.assertIsNotNone(ds.variables["u_random_btemps"])
        self.assertIsNotNone(ds.variables["u_non_random_btemps"])

    def testCreateTemplateFull_AMSUB(self):
        ds = FCDRWriter.createTemplateFull('AMSUB', 2562)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(29, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self._verify_amsub_specific_variables(ds)

        # full FCDR variables
        self.assertIsNotNone(ds.variables["u_btemps"])
        u_syst_btemps = ds.variables["u_syst_btemps"]
        self.assertIsNotNone(u_syst_btemps)
        u_random_btemps = ds.variables["u_random_btemps"]
        self.assertIsNotNone(u_random_btemps)
        u_instrtemp = ds.variables["u_instrtemp"]
        self.assertIsNotNone(u_instrtemp)
        u_latitude = ds.variables["u_latitude"]
        self.assertIsNotNone(u_latitude)
        u_longitude = ds.variables["u_longitude"]
        self.assertIsNotNone(u_longitude)
        u_satellite_azimuth_angle = ds.variables["u_satellite_azimuth_angle"]
        self.assertIsNotNone(u_satellite_azimuth_angle)
        u_satellite_zenith_angle = ds.variables["u_satellite_zenith_angle"]
        self.assertIsNotNone(u_satellite_zenith_angle)
        u_solar_azimuth_angle = ds.variables["u_solar_azimuth_angle"]
        self.assertIsNotNone(u_solar_azimuth_angle)
        u_solar_zenith_angle = ds.variables["u_solar_zenith_angle"]
        self.assertIsNotNone(u_solar_zenith_angle)

    def testCreateTemplateEasy_AVHRR(self):
        ds = FCDRWriter.createTemplateEasy('AVHRR', 12198)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(29, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self._verify_avhrr_specific_variables(ds)

        # easy FCDR variables
        self.assertIsNotNone(ds.variables["u_random_Ch1"])
        self.assertIsNotNone(ds.variables["u_non_random_Ch1"])
        self.assertIsNotNone(ds.variables["u_random_Ch2"])
        self.assertIsNotNone(ds.variables["u_non_random_Ch2"])
        self.assertIsNotNone(ds.variables["u_random_Ch3a"])
        self.assertIsNotNone(ds.variables["u_non_random_Ch3a"])
        self.assertIsNotNone(ds.variables["u_random_Ch3b"])
        self.assertIsNotNone(ds.variables["u_non_random_Ch3b"])
        self.assertIsNotNone(ds.variables["u_random_Ch4"])
        self.assertIsNotNone(ds.variables["u_non_random_Ch4"])
        self.assertIsNotNone(ds.variables["u_random_Ch5"])
        self.assertIsNotNone(ds.variables["u_non_random_Ch5"])

    def testCreateTemplateFull_AVHRR(self):
        ds = FCDRWriter.createTemplateFull('AVHRR', 13667)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(71, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self._verify_avhrr_specific_variables(ds)

        # variables of full FCDR
        self.assertIsNotNone(ds.variables["u_latitude"])
        self.assertIsNotNone(ds.variables["u_longitude"])
        self.assertIsNotNone(ds.variables["u_time"])
        self.assertIsNotNone(ds.variables["u_satellite_azimuth_angle"])
        self.assertIsNotNone(ds.variables["u_satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["u_solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["u_solar_zenith_angle"])

        self.assertIsNotNone(ds.variables["PRT_C"])
        self.assertIsNotNone(ds.variables["u_prt"])                             # geolocation
        self._verify_geolocation_variables(ds)
        self.assertIsNotNone(ds.variables["R_ICT"])
        self.assertIsNotNone(ds.variables["T_instr"])

        self.assertIsNotNone(ds.variables["Ch1_Csp"])
        self.assertIsNotNone(ds.variables["Ch2_Csp"])
        self.assertIsNotNone(ds.variables["Ch3a_Csp"])
        self.assertIsNotNone(ds.variables["Ch3b_Csp"])
        self.assertIsNotNone(ds.variables["Ch4_Csp"])
        self.assertIsNotNone(ds.variables["Ch5_Csp"])

        self.assertIsNotNone(ds.variables["Ch3b_Cict"])
        self.assertIsNotNone(ds.variables["Ch4_Cict"])
        self.assertIsNotNone(ds.variables["Ch5_Cict"])

        self.assertIsNotNone(ds.variables["Ch1_Ce"])
        self.assertIsNotNone(ds.variables["Ch2_Ce"])
        self.assertIsNotNone(ds.variables["Ch3a_Ce"])
        self.assertIsNotNone(ds.variables["Ch3b_Ce"])
        self.assertIsNotNone(ds.variables["Ch4_Ce"])
        self.assertIsNotNone(ds.variables["Ch5_Ce"])

        self.assertIsNotNone(ds.variables["Ch1_u_Csp"])
        self.assertIsNotNone(ds.variables["Ch2_u_Csp"])
        self.assertIsNotNone(ds.variables["Ch3a_u_Csp"])
        self.assertIsNotNone(ds.variables["Ch3b_u_Csp"])
        self.assertIsNotNone(ds.variables["Ch4_u_Csp"])
        self.assertIsNotNone(ds.variables["Ch5_u_Csp"])

        self.assertIsNotNone(ds.variables["Ch3b_u_Cict"])
        self.assertIsNotNone(ds.variables["Ch4_u_Cict"])
        self.assertIsNotNone(ds.variables["Ch5_u_Cict"])

        self.assertIsNotNone(ds.variables["Ch1_u_Ce"])
        self.assertIsNotNone(ds.variables["Ch2_u_Ce"])
        self.assertIsNotNone(ds.variables["Ch3a_u_Ce"])
        self.assertIsNotNone(ds.variables["Ch3b_u_Ce"])
        self.assertIsNotNone(ds.variables["Ch4_u_Ce"])
        self.assertIsNotNone(ds.variables["Ch5_u_Ce"])

        self.assertIsNotNone(ds.variables["Ch1_u_Refl"])
        self.assertIsNotNone(ds.variables["Ch2_u_Refl"])
        self.assertIsNotNone(ds.variables["Ch3a_u_Refl"])

        self.assertIsNotNone(ds.variables["Ch3b_u_Bt"])
        self.assertIsNotNone(ds.variables["Ch4_u_Bt"])
        self.assertIsNotNone(ds.variables["Ch5_u_Bt"])

        self.assertIsNotNone(ds.variables["Ch3b_ur_Bt"])
        self.assertIsNotNone(ds.variables["Ch4_ur_Bt"])
        self.assertIsNotNone(ds.variables["Ch5_ur_Bt"])

        self.assertIsNotNone(ds.variables["Ch3b_us_Bt"])
        self.assertIsNotNone(ds.variables["Ch4_us_Bt"])
        self.assertIsNotNone(ds.variables["Ch5_us_Bt"])

    def testCreateTemplateEasy_HIRS2(self):
        ds = FCDRWriter.createTemplateEasy('HIRS2', 211)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(16, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self.assertIsNotNone(ds.variables["bt"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["scanline"])
        self.assertIsNotNone(ds.variables["time"])
        self.assertIsNotNone(ds.variables["scnlintime"])
        self.assertIsNotNone(ds.variables["scnlinf"])
        self.assertIsNotNone(ds.variables["scantype"])
        self.assertIsNotNone(ds.variables["qualind"])

        # easy FCDR variables
        self.assertIsNotNone(ds.variables["u_random"])
        self.assertIsNotNone(ds.variables["u_non_random"])

    def testCreateTemplateFull_HIRS2(self):
        ds = FCDRWriter.createTemplateFull('HIRS2', 209)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(137, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # TODO 1 tb/tb 2017-03-08 ad more sensor variables, maybe extract common assert method
        self.assertIsNotNone(ds.variables["bt"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["scanline"])
        self.assertIsNotNone(ds.variables["scnlintime"])
        self.assertIsNotNone(ds.variables["scantype"])
        self.assertIsNotNone(ds.variables["qualind"])

        self.assertIsNotNone(ds.variables["c_earth"])
        self.assertIsNotNone(ds.variables["L_earth"])

        self.assertIsNotNone(ds.variables["navigation_status"])
        self.assertIsNotNone(ds.variables["platform_altitude"])
        self.assertIsNotNone(ds.variables["platform_pitch_angle"])
        self.assertIsNotNone(ds.variables["platform_roll_angle"])
        self.assertIsNotNone(ds.variables["platform_yaw_angle"])
        self.assertIsNotNone(ds.variables["quality_flags"])
        self.assertIsNotNone(ds.variables["scan_angles"])
        self.assertIsNotNone(ds.variables["scanline_number"])
        self.assertIsNotNone(ds.variables["scanline_position"])
        self.assertIsNotNone(ds.variables["second_original_calibration_coefficients"])
        self.assertIsNotNone(ds.variables["TK_baseplate"])
        self.assertIsNotNone(ds.variables["TK_baseplate_analog"])
        self.assertIsNotNone(ds.variables["TK_ch"])
        self.assertIsNotNone(ds.variables["TK_elec_analog"])
        self.assertIsNotNone(ds.variables["TK_elec"])
        self.assertIsNotNone(ds.variables["TK_fwm_analog"])
        self.assertIsNotNone(ds.variables["TK_fwh"])
        self.assertIsNotNone(ds.variables["TK_iwct"])
        self.assertIsNotNone(ds.variables["TK_patch_analog"])
        self.assertIsNotNone(ds.variables["TK_patch_exp"])
        self.assertIsNotNone(ds.variables["TK_patch_full"])
        self.assertIsNotNone(ds.variables["TK_radiator_analog"])
        self.assertIsNotNone(ds.variables["TK_scanmirror"])
        self.assertIsNotNone(ds.variables["TK_scanmirror_analog"])
        self.assertIsNotNone(ds.variables["TK_tlscp_sec"])
        self.assertIsNotNone(ds.variables["TK_scanmotor"])
        self.assertIsNotNone(ds.variables["u_c_earth"])
        self.assertIsNotNone(ds.variables["u_c_earth_chan_corr"])
        self.assertIsNotNone(ds.variables["u_c_space"])
        self.assertIsNotNone(ds.variables["u_c_space_chan_corr"])
        self.assertIsNotNone(ds.variables["u_Earthshine"])
        self.assertIsNotNone(ds.variables["u_O_Re"])
        self.assertIsNotNone(ds.variables["u_O_TIWCT"])
        self.assertIsNotNone(ds.variables["u_O_TPRT"])
        self.assertIsNotNone(ds.variables["u_O_TPRT_chan_corr"])
        self.assertIsNotNone(ds.variables["u_Rself"])
        self.assertIsNotNone(ds.variables["u_Rselfparams"])
        self.assertIsNotNone(ds.variables["u_SRF_calib"])
        self.assertIsNotNone(ds.variables["u_d_PRT"])
        self.assertIsNotNone(ds.variables["u_electronics"])
        self.assertIsNotNone(ds.variables["u_extraneous_periodic"])
        self.assertIsNotNone(ds.variables["u_nonlinearity"])

        self.assertIsNotNone(ds.variables["temp_corr_slope"])
        self.assertIsNotNone(ds.variables["temp_corr_offset"])
        self.assertIsNotNone(ds.variables["emissivity"])

    def testCreateTemplateEasy_HIRS3(self):
        ds = FCDRWriter.createTemplateEasy('HIRS3', 211)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(22, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self.assertIsNotNone(ds.variables["bt"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["satellite_azimuth_angle"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["scanline"])
        self.assertIsNotNone(ds.variables["time"])
        self.assertIsNotNone(ds.variables["scnlintime"])
        self.assertIsNotNone(ds.variables["scnlinf"])
        self.assertIsNotNone(ds.variables["scantype"])
        self.assertIsNotNone(ds.variables["linqualflags"])
        self.assertIsNotNone(ds.variables["chqualflags"])
        self.assertIsNotNone(ds.variables["mnfrqualflags"])

        # easy FCDR variables
        self.assertIsNotNone(ds.variables["u_random"])
        self.assertIsNotNone(ds.variables["u_non_random"])

    def testCreateTemplateFull_HIRS3(self):
        ds = FCDRWriter.createTemplateFull('HIRS3', 209)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(143, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # TODO 1 tb/tb 2017-03-08 ad more sensor variables, maybe extract common assert method
        self.assertIsNotNone(ds.variables["bt"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["scnlintime"])
        self.assertIsNotNone(ds.variables["scanline"])
        self.assertIsNotNone(ds.variables["scantype"])

        self.assertIsNotNone(ds.variables["c_earth"])
        self.assertIsNotNone(ds.variables["L_earth"])

        self.assertIsNotNone(ds.variables["navigation_status"])
        self.assertIsNotNone(ds.variables["platform_altitude"])
        self.assertIsNotNone(ds.variables["platform_pitch_angle"])
        self.assertIsNotNone(ds.variables["platform_roll_angle"])
        self.assertIsNotNone(ds.variables["platform_yaw_angle"])
        self.assertIsNotNone(ds.variables["quality_flags"])
        self.assertIsNotNone(ds.variables["scan_angles"])
        self.assertIsNotNone(ds.variables["scanline_number"])
        self.assertIsNotNone(ds.variables["scanline_position"])
        self.assertIsNotNone(ds.variables["second_original_calibration_coefficients"])
        self.assertIsNotNone(ds.variables["TK_baseplate"])
        self.assertIsNotNone(ds.variables["TK_baseplate_analog"])
        self.assertIsNotNone(ds.variables["TK_ch"])
        self.assertIsNotNone(ds.variables["TK_elec_analog"])
        self.assertIsNotNone(ds.variables["TK_elec"])
        self.assertIsNotNone(ds.variables["TK_fwm_analog"])
        self.assertIsNotNone(ds.variables["TK_fwh"])
        self.assertIsNotNone(ds.variables["TK_iwct"])
        self.assertIsNotNone(ds.variables["TK_patch_analog"])
        self.assertIsNotNone(ds.variables["TK_patch_exp"])
        self.assertIsNotNone(ds.variables["TK_patch_full"])
        self.assertIsNotNone(ds.variables["TK_radiator_analog"])
        self.assertIsNotNone(ds.variables["TK_scanmirror"])
        self.assertIsNotNone(ds.variables["TK_scanmirror_analog"])
        self.assertIsNotNone(ds.variables["TK_tlscp_sec"])
        self.assertIsNotNone(ds.variables["TK_scanmotor"])
        self.assertIsNotNone(ds.variables["u_c_earth"])
        self.assertIsNotNone(ds.variables["u_c_earth_chan_corr"])
        self.assertIsNotNone(ds.variables["u_c_space"])
        self.assertIsNotNone(ds.variables["u_c_space_chan_corr"])
        self.assertIsNotNone(ds.variables["u_Earthshine"])
        self.assertIsNotNone(ds.variables["u_O_Re"])
        self.assertIsNotNone(ds.variables["u_O_TIWCT"])
        self.assertIsNotNone(ds.variables["u_O_TPRT"])
        self.assertIsNotNone(ds.variables["u_O_TPRT_chan_corr"])
        self.assertIsNotNone(ds.variables["u_Rself"])
        self.assertIsNotNone(ds.variables["u_Rselfparams"])
        self.assertIsNotNone(ds.variables["u_SRF_calib"])
        self.assertIsNotNone(ds.variables["u_d_PRT"])
        self.assertIsNotNone(ds.variables["u_electronics"])
        self.assertIsNotNone(ds.variables["u_extraneous_periodic"])
        self.assertIsNotNone(ds.variables["u_nonlinearity"])

        self.assertIsNotNone(ds.variables["temp_corr_slope"])
        self.assertIsNotNone(ds.variables["temp_corr_offset"])
        self.assertIsNotNone(ds.variables["emissivity"])

    def testCreateTemplateEasy_HIRS4(self):
        ds = FCDRWriter.createTemplateEasy('HIRS4', 211)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(22, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self.assertIsNotNone(ds.variables["bt"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["satellite_azimuth_angle"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["scanline"])
        self.assertIsNotNone(ds.variables["time"])
        self.assertIsNotNone(ds.variables["scnlintime"])
        self.assertIsNotNone(ds.variables["scnlinf"])
        self.assertIsNotNone(ds.variables["scantype"])
        self.assertIsNotNone(ds.variables["linqualflags"])
        self.assertIsNotNone(ds.variables["chqualflags"])
        self.assertIsNotNone(ds.variables["mnfrqualflags"])

        # easy FCDR variables
        self.assertIsNotNone(ds.variables["u_random"])
        self.assertIsNotNone(ds.variables["u_non_random"])

    def testCreateTemplateFull_HIRS4(self):
        ds = FCDRWriter.createTemplateFull('HIRS4', 209)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(143, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # TODO 1 tb/tb 2017-03-08 ad more sensor variables, maybe extract common assert method
        self.assertIsNotNone(ds.variables["bt"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["scanline"])
        self.assertIsNotNone(ds.variables["scnlintime"])
        self.assertIsNotNone(ds.variables["scantype"])

        self.assertIsNotNone(ds.variables["c_earth"])
        self.assertIsNotNone(ds.variables["L_earth"])

        self.assertIsNotNone(ds.variables["navigation_status"])
        self.assertIsNotNone(ds.variables["platform_altitude"])
        self.assertIsNotNone(ds.variables["platform_pitch_angle"])
        self.assertIsNotNone(ds.variables["platform_roll_angle"])
        self.assertIsNotNone(ds.variables["platform_yaw_angle"])
        self.assertIsNotNone(ds.variables["quality_flags"])
        self.assertIsNotNone(ds.variables["scan_angles"])
        self.assertIsNotNone(ds.variables["scanline_number"])
        self.assertIsNotNone(ds.variables["scanline_position"])
        self.assertIsNotNone(ds.variables["second_original_calibration_coefficients"])
        self.assertIsNotNone(ds.variables["TK_baseplate"])
        self.assertIsNotNone(ds.variables["TK_baseplate_analog"])
        self.assertIsNotNone(ds.variables["TK_ch"])
        self.assertIsNotNone(ds.variables["TK_elec_analog"])
        self.assertIsNotNone(ds.variables["TK_elec"])
        self.assertIsNotNone(ds.variables["TK_fwm_analog"])
        self.assertIsNotNone(ds.variables["TK_fwh"])
        self.assertIsNotNone(ds.variables["TK_iwct"])
        self.assertIsNotNone(ds.variables["TK_patch_analog"])
        self.assertIsNotNone(ds.variables["TK_patch_exp"])
        self.assertIsNotNone(ds.variables["TK_patch_full"])
        self.assertIsNotNone(ds.variables["TK_radiator_analog"])
        self.assertIsNotNone(ds.variables["TK_scanmirror"])
        self.assertIsNotNone(ds.variables["TK_scanmirror_analog"])
        self.assertIsNotNone(ds.variables["TK_tlscp_sec"])
        self.assertIsNotNone(ds.variables["TK_scanmotor"])
        self.assertIsNotNone(ds.variables["u_c_earth"])
        self.assertIsNotNone(ds.variables["u_c_earth_chan_corr"])
        self.assertIsNotNone(ds.variables["u_c_space"])
        self.assertIsNotNone(ds.variables["u_c_space_chan_corr"])
        self.assertIsNotNone(ds.variables["u_Earthshine"])
        self.assertIsNotNone(ds.variables["u_O_Re"])
        self.assertIsNotNone(ds.variables["u_O_TIWCT"])
        self.assertIsNotNone(ds.variables["u_O_TPRT"])
        self.assertIsNotNone(ds.variables["u_O_TPRT_chan_corr"])
        self.assertIsNotNone(ds.variables["u_Rself"])
        self.assertIsNotNone(ds.variables["u_Rselfparams"])
        self.assertIsNotNone(ds.variables["u_SRF_calib"])
        self.assertIsNotNone(ds.variables["u_d_PRT"])
        self.assertIsNotNone(ds.variables["u_electronics"])
        self.assertIsNotNone(ds.variables["u_extraneous_periodic"])
        self.assertIsNotNone(ds.variables["u_nonlinearity"])

        self.assertIsNotNone(ds.variables["temp_corr_slope"])
        self.assertIsNotNone(ds.variables["temp_corr_offset"])
        self.assertIsNotNone(ds.variables["emissivity"])

    def testCreateTemplateEasy_MVIRI(self):
        ds = FCDRWriter.createTemplateEasy('MVIRI', 5000)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(36, len(ds.variables))

        # sensor specific
        self.assertIsNotNone(ds.variables["time"])
        self.assertIsNotNone(ds.variables["toa_bidirectional_reflectance"])
        self.assertIsNotNone(ds.variables["spectral_response_function_vis"])
        self.assertIsNotNone(ds.variables["covariance_spectral_response_function_vis"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["satellite_azimuth_angle"])
        self.assertIsNotNone(ds.variables["count_ir"])
        self.assertIsNotNone(ds.variables["count_wv"])
        self.assertIsNotNone(ds.variables["distance_sun_earth"])
        self.assertIsNotNone(ds.variables["sol_eff_irr"])
        self.assertIsNotNone(ds.variables["u_sol_eff_irr"])
        self.assertIsNotNone(ds.variables["a_ir"])
        self.assertIsNotNone(ds.variables["b_ir"])
        self.assertIsNotNone(ds.variables["u_a_ir"])
        self.assertIsNotNone(ds.variables["u_b_ir"])
        self.assertIsNotNone(ds.variables["a_wv"])
        self.assertIsNotNone(ds.variables["b_wv"])
        self.assertIsNotNone(ds.variables["u_a_wv"])
        self.assertIsNotNone(ds.variables["u_b_wv"])
        self.assertIsNotNone(ds.variables["q_ir"])
        self.assertIsNotNone(ds.variables["q_wv"])
        self.assertIsNotNone(ds.variables["unit_conversion_ir"])
        self.assertIsNotNone(ds.variables["unit_conversion_wv"])
        self.assertIsNotNone(ds.variables["bt_a_ir"])
        self.assertIsNotNone(ds.variables["bt_b_ir"])
        self.assertIsNotNone(ds.variables["bt_a_wv"])
        self.assertIsNotNone(ds.variables["bt_b_wv"])

        # easy FCDR uncertainties
        self.assertIsNotNone(ds.variables["u_random"])
        self.assertIsNotNone(ds.variables["u_non_random"])

    def testCreateTemplateFull_MVIRI(self):
        ds = FCDRWriter.createTemplateFull('MVIRI', 5000)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(52, len(ds.variables))

        # sensor specific
        self.assertIsNotNone(ds.variables["count_vis"])
        self.assertIsNotNone(ds.variables["time"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["satellite_azimuth_angle"])
        self.assertIsNotNone(ds.variables["count_ir"])
        self.assertIsNotNone(ds.variables["count_wv"])
        self.assertIsNotNone(ds.variables["count_vis"])
        self.assertIsNotNone(ds.variables["spectral_response_function_vis"])
        self.assertIsNotNone(ds.variables["covariance_spectral_response_function_vis"])
        self.assertIsNotNone(ds.variables["a0_vis"])
        self.assertIsNotNone(ds.variables["a1_vis"])
        self.assertIsNotNone(ds.variables["sol_eff_irr"])
        self.assertIsNotNone(ds.variables["distance_sun_earth"])
        self.assertIsNotNone(ds.variables["mean_counts_space_vis"])
        self.assertIsNotNone(ds.variables["a_ir"])
        self.assertIsNotNone(ds.variables["b_ir"])
        self.assertIsNotNone(ds.variables["u_a_ir"])
        self.assertIsNotNone(ds.variables["u_b_ir"])
        self.assertIsNotNone(ds.variables["a_wv"])
        self.assertIsNotNone(ds.variables["b_wv"])
        self.assertIsNotNone(ds.variables["u_a_wv"])
        self.assertIsNotNone(ds.variables["u_b_wv"])
        self.assertIsNotNone(ds.variables["q_ir"])
        self.assertIsNotNone(ds.variables["q_wv"])
        self.assertIsNotNone(ds.variables["unit_conversion_ir"])
        self.assertIsNotNone(ds.variables["unit_conversion_wv"])
        self.assertIsNotNone(ds.variables["bt_a_ir"])
        self.assertIsNotNone(ds.variables["bt_b_ir"])
        self.assertIsNotNone(ds.variables["bt_a_wv"])
        self.assertIsNotNone(ds.variables["bt_b_wv"])

        # full FCDR uncertainties
        self.assertIsNotNone(ds.variables["u_latitude"])
        self.assertIsNotNone(ds.variables["u_longitude"])
        self.assertIsNotNone(ds.variables["u_time"])
        self.assertIsNotNone(ds.variables["u_combined_counts_vis"])
        self.assertIsNotNone(ds.variables["u_a0_vis"])
        self.assertIsNotNone(ds.variables["u_a1_vis"])
        self.assertIsNotNone(ds.variables["covariance_a0_a1_vis"])
        self.assertIsNotNone(ds.variables["u_sol_eff_irr"])
        self.assertIsNotNone(ds.variables["u_electronics_counts_vis"])
        self.assertIsNotNone(ds.variables["u_digitization_counts_vis"])
        self.assertIsNotNone(ds.variables["allan_deviation_counts_space_vis"])
        self.assertIsNotNone(ds.variables["u_solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["u_solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["u_satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["u_satellite_azimuth_angle"])

    def _verifyGlobalAttributes(self, attributes):
        self.assertIsNotNone(attributes)
        self.assertEqual("CF-1.6", attributes["Conventions"])
        self.assertEqual(
            "This dataset is released for use under CC-BY licence (https://creativecommons.org/licenses/by/4.0/) and was developed in the EC "
            "FIDUCEO project \"Fidelity and Uncertainty in Climate Data Records from Earth "
            "Observations\". Grant Agreement: 638822.", attributes["licence"])
        self.assertEqual("1.0.7", attributes["writer_version"])

    def _verify_geolocation_variables(self, ds):
        self.assertIsNotNone(ds.variables["latitude"])
        self.assertIsNotNone(ds.variables["longitude"])

    def _verify_geometry_variables(self, ds):
        self.assertIsNotNone(ds.variables["satellite_azimuth_angle"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])

    def _verify_amsub_specific_variables(self, ds):
        btemps = ds.variables["btemps"]
        self.assertIsNotNone(btemps)
        chanqual = ds.variables["chanqual"]
        self.assertIsNotNone(chanqual)
        instrtemp = ds.variables["instrtemp"]
        self.assertIsNotNone(instrtemp)
        qualind = ds.variables["qualind"]
        self.assertIsNotNone(qualind)
        scanqual = ds.variables["scanqual"]
        self.assertIsNotNone(scanqual)
        scnlin = ds.variables["scnlin"]
        self.assertIsNotNone(scnlin)
        scnlindy = ds.variables["scnlindy"]
        self.assertIsNotNone(scnlindy)
        scnlintime = ds.variables["scnlintime"]
        self.assertIsNotNone(scnlintime)
        scnlinyr = ds.variables["scnlinyr"]
        self.assertIsNotNone(scnlinyr)
        # geometry
        self._verify_geometry_variables(ds)

    def _verify_avhrr_specific_variables(self, ds):
        self.assertIsNotNone(ds.variables["Time"])
        self.assertIsNotNone(ds.variables["scanline"])
        # geometry
        self._verify_geometry_variables(ds)

        self.assertIsNotNone(ds.variables["Ch1_Ref"])
        self.assertIsNotNone(ds.variables["Ch2_Ref"])
        self.assertIsNotNone(ds.variables["Ch3a_Ref"])
        self.assertIsNotNone(ds.variables["Ch3b_Bt"])
        self.assertIsNotNone(ds.variables["Ch4_Bt"])
        self.assertIsNotNone(ds.variables["Ch5_Bt"])
        self.assertIsNotNone(ds.variables["T_ICT"])
