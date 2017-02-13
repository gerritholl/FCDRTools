import unittest

from writer.fcdr_writer import FCDRWriter


class FCDRWriterTest(unittest.TestCase):
    def testCreateTemplateEasy_AMSUB(self):
        ds = FCDRWriter.createTemplateEasy('AMSUB', 2561)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(19, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self._verify_amsub_specific_variables(ds)

        # easy FCDR variables
        # TODO 1 tb/tb 2017-02-13

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
        u_btemps = ds.variables["u_btemps"]
        self.assertIsNotNone(u_btemps)
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

        self.assertEqual(17, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self._verify_avhrr_specific_variables(ds)

        # easy FCDR variables
        # TODO 1 tb/tb 2017-02-13

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
        self.assertIsNotNone(ds.variables["u_prt"])
        self.assertIsNotNone(ds.variables["R_ICT"])
        self.assertIsNotNone(ds.variables["T_instr"])

        self.assertIsNotNone(ds.variables["Ch1_Csp"])
        self.assertIsNotNone(ds.variables["Ch2_Csp"])
        self.assertIsNotNone(ds.variables["Ch3a_Csp"])
        self.assertIsNotNone(ds.variables["Ch3b_Csp"])
        self.assertIsNotNone(ds.variables["Ch4_Csp"])
        self.assertIsNotNone(ds.variables["Ch5_Csp"])

        # @todo 2 tb/tb continue here 2017-01-27

    def testCreateTemplateEasy_HIRS(self):
        ds = FCDRWriter.createTemplateEasy('HIRS', 211)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(21, len(ds.variables))

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        self.assertIsNotNone(ds.variables["bt"])
        self.assertIsNotNone(ds.variables["c_earth"])
        self.assertIsNotNone(ds.variables["L_earth"])
        self.assertIsNotNone(ds.variables["sat_za"])
        self.assertIsNotNone(ds.variables["sat_aa"])
        self.assertIsNotNone(ds.variables["sol_za"])
        self.assertIsNotNone(ds.variables["sol_aa"])
        self.assertIsNotNone(ds.variables["scanline"])
        self.assertIsNotNone(ds.variables["time"])
        self.assertIsNotNone(ds.variables["scnlinf"])
        self.assertIsNotNone(ds.variables["linqualflags"])
        self.assertIsNotNone(ds.variables["chqualflags"])
        self.assertIsNotNone(ds.variables["mnfrqualflags"])

        # easy FCDR variables
        # TODO 1 tb/tb 2017-02-13

    def testCreateTemplateEasy_MVIRI(self):
        ds = FCDRWriter.createTemplateEasy('MVIRI', 5000)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(15, len(ds.variables))

        # sensor specific
        self.assertIsNotNone(ds.variables["time"])
        self.assertIsNotNone(ds.variables["timedelta"])
        self.assertIsNotNone(ds.variables["reflectance"])
        self.assertIsNotNone(ds.variables["srf"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["satellite_azimuth_angle"])

        # easy FCDR uncertainties
        self.assertIsNotNone(ds.variables["u_random"])
        self.assertIsNotNone(ds.variables["u_non_random"])

    def testCreateTemplateFull_MVIRI(self):
        ds = FCDRWriter.createTemplateFull('MVIRI', 5000)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        self.assertEqual(37, len(ds.variables))

        # sensor specific
        self.assertIsNotNone(ds.variables["time"])
        self.assertIsNotNone(ds.variables["timedelta"])
        self.assertIsNotNone(ds.variables["solar_zenith_angle"])
        self.assertIsNotNone(ds.variables["solar_azimuth_angle"])
        self.assertIsNotNone(ds.variables["satellite_zenith_angle"])
        self.assertIsNotNone(ds.variables["satellite_azimuth_angle"])
        self.assertIsNotNone(ds.variables["count"])
        self.assertIsNotNone(ds.variables["reflectance"])
        self.assertIsNotNone(ds.variables["srf"])
        self.assertIsNotNone(ds.variables["a0"])
        self.assertIsNotNone(ds.variables["a1"])
        self.assertIsNotNone(ds.variables["sol_irr"])
        self.assertIsNotNone(ds.variables["sol_eff_irr"])
        self.assertIsNotNone(ds.variables["dSE"])
        self.assertIsNotNone(ds.variables["K_space"])

        # full FCDR uncertainties
        self.assertIsNotNone(ds.variables["u_latitude"])
        self.assertIsNotNone(ds.variables["u_longitude"])
        self.assertIsNotNone(ds.variables["u_time"])
        self.assertIsNotNone(ds.variables["u_tot_count"])
        self.assertIsNotNone(ds.variables["u_srf"])
        self.assertIsNotNone(ds.variables["u_a0"])
        self.assertIsNotNone(ds.variables["u_a1"])
        self.assertIsNotNone(ds.variables["u_sol_eff_irr"])
        self.assertIsNotNone(ds.variables["u_dSE"])
        self.assertIsNotNone(ds.variables["u_e-noise"])
        self.assertIsNotNone(ds.variables["u_shot-noise"])
        self.assertIsNotNone(ds.variables["u_digitization"])
        self.assertIsNotNone(ds.variables["u_space"])
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

    def _verify_geolocation_variables(self, ds):
        latitude = ds.variables["latitude"]
        self.assertIsNotNone(latitude)
        longitude = ds.variables["longitude"]
        self.assertIsNotNone(longitude)

    def _verify_geometry_variables(self, ds):
        sat_azimuth = ds.variables["satellite_azimuth_angle"]
        self.assertIsNotNone(sat_azimuth)
        sat_zenith = ds.variables["satellite_zenith_angle"]
        self.assertIsNotNone(sat_zenith)
        sol_azimuth = ds.variables["solar_azimuth_angle"]
        self.assertIsNotNone(sol_azimuth)
        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertIsNotNone(sol_zenith)

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
        time = ds.variables["Time"]
        self.assertIsNotNone(time)
        scanline = ds.variables["scanline"]
        self.assertIsNotNone(scanline)
        # geometry
        self._verify_geometry_variables(ds)
        ch1_bt = ds.variables["Ch1_Bt"]
        self.assertIsNotNone(ch1_bt)
        ch2_bt = ds.variables["Ch2_Bt"]
        self.assertIsNotNone(ch2_bt)
        ch3a_bt = ds.variables["Ch3a_Bt"]
        self.assertIsNotNone(ch3a_bt)
        ch3b_bt = ds.variables["Ch3b_Bt"]
        self.assertIsNotNone(ch3b_bt)
        ch4_bt = ds.variables["Ch4_Bt"]
        self.assertIsNotNone(ch4_bt)
        ch5_bt = ds.variables["Ch5_Bt"]
        self.assertIsNotNone(ch5_bt)
        t_ict = ds.variables["T_ICT"]
        self.assertIsNotNone(t_ict)
