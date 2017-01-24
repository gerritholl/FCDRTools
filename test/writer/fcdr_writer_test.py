import unittest

from writer.fcdr_writer import FCDRWriter


class FCDRWriterTest(unittest.TestCase):
    def testCreateTemplateEasy_AMSUB(self):
        ds = FCDRWriter.createTemplateEasy('AMSUB', 2561)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
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

        # easy FCDR variables
        self._verify_easy_fcdr_variables(ds)

    def testCreateTemplateEasy_AVHRR(self):
        ds = FCDRWriter.createTemplateEasy('AVHRR', 12198)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
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

        # easy FCDR variables
        self._verify_easy_fcdr_variables(ds)

    def testCreateTemplateEasy_HIRS(self):
        ds = FCDRWriter.createTemplateEasy('HIRS', 211)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        bt = ds.variables["bt"]
        self.assertIsNotNone(bt)
        c_earth = ds.variables["c_earth"]
        self.assertIsNotNone(c_earth)
        l_earth = ds.variables["L_earth"]
        self.assertIsNotNone(l_earth)
        sat_za = ds.variables["sat_za"]
        self.assertIsNotNone(sat_za)
        scanline = ds.variables["scanline"]
        self.assertIsNotNone(scanline)
        scnlinf = ds.variables["scnlinf"]
        self.assertIsNotNone(scnlinf)

        # easy FCDR variables
        self._verify_easy_fcdr_variables(ds)

    def testCreateTemplateEasy_MVIRI(self):
        ds = FCDRWriter.createTemplateEasy('MVIRI', 4000)
        self.assertIsNotNone(ds)

        self._verifyGlobalAttributes(ds.attrs)

        # geolocation
        self._verify_geolocation_variables(ds)

        # sensor specific
        time = ds.variables["time"]
        self.assertIsNotNone(time)
        time_delta = ds.variables["time_delta"]
        self.assertIsNotNone(time_delta)
        # geometry
        self._verify_geometry_variables(ds)
        count = ds.variables["count"]
        self.assertIsNotNone(count)

        # easy FCDR variables
        self._verify_easy_fcdr_variables(ds)

    def _verifyGlobalAttributes(self, attributes):
        self.assertIsNotNone(attributes)
        self.assertEqual("CF-1.6", attributes["Conventions"])
        self.assertEqual("This dataset is released for use under CC-BY licence and was developed in the EC "
                         "FIDUCEO project \"Fidelity and Uncertainty in Climate Data Records from Earth "
                         "Observations\". Grant Agreement: 638822.", attributes["license"])

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

    def _verify_easy_fcdr_variables(self, ds):
        u_random = ds.variables["u_random"]
        self.assertIsNotNone(u_random)
        u_systematic = ds.variables["u_systematic"]
        self.assertIsNotNone(u_systematic)
