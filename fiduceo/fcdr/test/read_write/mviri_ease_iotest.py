import os
import tempfile
import unittest
from datetime import datetime

import numpy as np
import xarray as xr

from fiduceo.fcdr.writer.fcdr_writer import FCDRWriter

EXPECTED_CHUNKING = (512, 512)


class MviriEASYIoTest(unittest.TestCase):
    temp_dir = None
    target_path = None

    def setUp(self):
        self.temp_dir = tempfile.gettempdir()
        self.target_path = None

    def tearDown(self):
        if self.target_path is not None:
            os.remove(self.target_path)

    def test_write(self):
        mviri_easy = self.create_easy_dataset()

        start = datetime(2011, 9, 12, 13, 24, 52)
        end = datetime(2011, 9, 12, 13, 27, 51)
        file_name = FCDRWriter.create_file_name_FCDR_easy("MVIRI", "Meteosat8", start, end, "1.0")
        self.target_path = os.path.join(self.temp_dir, file_name)

        start_time = datetime.now()

        FCDRWriter.write(mviri_easy, self.target_path)

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print("MVIRI EASY write time: " + str(elapsed_time.seconds) + "." + str(round(elapsed_time.microseconds / 1000)))

        self.assertTrue(os.path.isfile(self.target_path))

        target_data = xr.open_dataset(self.target_path)
        try:
            variable = target_data["time"]
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["solar_azimuth_angle"]
            self.assertAlmostEqual(0.010986328, variable.data[1, 1], 8)
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["solar_zenith_angle"]
            self.assertAlmostEqual(0.038452736, variable.data[2, 2], 8)
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["count_ir"]
            self.assertEqual(3, variable.data[3, 3])
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["count_wv"]
            self.assertEqual(5, variable.data[4, 4])
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["distance_sun_earth"]
            self.assertAlmostEqual(3.3, variable.data, 8)

            variable = target_data["solar_irradiance_vis"]
            self.assertAlmostEqual(4.4, variable.data, 8)

            variable = target_data["u_solar_irradiance_vis"]
            self.assertAlmostEqual(5.5, variable.data, 8)

            variable = target_data["spectral_response_function_vis"]
            self.assertAlmostEqual(0.03, variable.data[4], 8)
            self.assertEqual((1011,), variable.encoding["chunksizes"])

            variable = target_data["covariance_spectral_response_function_vis"]
            self.assertAlmostEqual(40.4, variable.data[5, 5], 5)
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["spectral_response_function_ir"]
            self.assertAlmostEqual(0.04, variable.data[6], 8)
            self.assertEqual((1011,), variable.encoding["chunksizes"])

            variable = target_data["u_spectral_response_function_ir"]
            self.assertAlmostEqual(0.05, variable.data[7], 8)
            self.assertEqual((1011,), variable.encoding["chunksizes"])

            variable = target_data["spectral_response_function_wv"]
            self.assertAlmostEqual(0.06, variable.data[8], 8)
            self.assertEqual((1011,), variable.encoding["chunksizes"])

            variable = target_data["u_spectral_response_function_wv"]
            self.assertAlmostEqual(0.07, variable.data[9], 8)
            self.assertEqual((1011,), variable.encoding["chunksizes"])

            variable = target_data["a_ir"]
            self.assertAlmostEqual(8.8, variable.data, 8)

            variable = target_data["b_ir"]
            self.assertAlmostEqual(9.9, variable.data, 8)

            variable = target_data["u_a_ir"]
            self.assertAlmostEqual(10.1, variable.data, 8)

            variable = target_data["u_b_ir"]
            self.assertAlmostEqual(11.11, variable.data, 8)

            variable = target_data["a_wv"]
            self.assertAlmostEqual(12.12, variable.data, 8)

            variable = target_data["b_wv"]
            self.assertAlmostEqual(13.13, variable.data, 8)

            variable = target_data["u_a_wv"]
            self.assertAlmostEqual(14.14, variable.data, 8)

            variable = target_data["u_b_wv"]
            self.assertAlmostEqual(15.15, variable.data, 8)

            variable = target_data["q_ir"]
            self.assertAlmostEqual(16.16, variable.data, 8)

            variable = target_data["q_wv"]
            self.assertAlmostEqual(17.17, variable.data, 8)

            variable = target_data["unit_conversion_ir"]
            self.assertAlmostEqual(18.18, variable.data, 8)

            variable = target_data["unit_conversion_wv"]
            self.assertAlmostEqual(19.19, variable.data, 8)

            variable = target_data["bt_a_ir"]
            self.assertAlmostEqual(20.2, variable.data, 8)

            variable = target_data["bt_b_ir"]
            self.assertAlmostEqual(21.21, variable.data, 8)

            variable = target_data["bt_a_wv"]
            self.assertAlmostEqual(22.22, variable.data, 8)

            variable = target_data["bt_b_wv"]
            self.assertAlmostEqual(23.23, variable.data, 8)

            variable = target_data["years_since_launch"]
            self.assertAlmostEqual(24.24, variable.data, 8)

            variable = target_data["toa_bidirectional_reflectance_vis"]
            self.assertAlmostEqual(0.75, variable.data[25, 25], 5)
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["u_independent_toa_bidirectional_reflectance"]
            self.assertAlmostEqual(0.04, variable.data[26, 26], 4)
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["u_structured_toa_bidirectional_reflectance"]
            self.assertAlmostEqual(0.35, variable.data[27, 27], 4)
            self.assertEqual(EXPECTED_CHUNKING, variable.encoding["chunksizes"])

            variable = target_data["sub_satellite_latitude_start"]
            self.assertAlmostEqual(25.25, variable.data, 8)

            variable = target_data["sub_satellite_longitude_start"]
            self.assertAlmostEqual(26.26, variable.data, 8)

            variable = target_data["sub_satellite_latitude_end"]
            self.assertAlmostEqual(27.27, variable.data, 8)

            variable = target_data["sub_satellite_longitude_end"]
            self.assertAlmostEqual(28.28, variable.data, 8)
        finally:
            target_data.close()

    def create_easy_dataset(self):
        mviri_easy = FCDRWriter.createTemplateEasy("MVIRI", 5000)
        mviri_easy.attrs["institution"] = "test"
        mviri_easy.attrs["title"] = "sir"
        mviri_easy.attrs["source"] = "invention"
        mviri_easy.attrs["history"] = "new"
        mviri_easy.attrs["references"] = "myself"
        mviri_easy.attrs["comment"] = "should define a test version of this set"

        for x in range(0, 2500):
            mviri_easy["time"].data[:, x] = np.ones((2500), np.uint32) * x
            mviri_easy["count_ir"].data[:, x] = np.ones((2500), np.uint8) * x
            mviri_easy["count_wv"].data[:, x] = np.ones((2500), np.uint8) * x + 1

        for x in range(0, 5000):
            mviri_easy["solar_azimuth_angle"].data[:, x] = np.ones((5000), np.uint32) * x * 0.01
            mviri_easy["solar_zenith_angle"].data[:, x] = np.ones((5000), np.uint32) * x * 0.02
            mviri_easy["toa_bidirectional_reflectance_vis"].data[:, x] = np.ones((5000), np.uint16) * x * 0.03
            mviri_easy["u_independent_toa_bidirectional_reflectance"].data[:, x] = np.ones((5000), np.uint16) * x * 0.04
            mviri_easy["u_structured_toa_bidirectional_reflectance"].data[:, x] = np.ones((5000), np.uint16) * x * 0.05

        for x in range(0, 1011):
            mviri_easy["covariance_spectral_response_function_vis"].data[:] = np.ones((1011), np.float32) * x * 0.04

        # for x in range(0, 6):
        #     mviri_easy["quality_channel_bitmask"].data[:, x] = np.ones((13198), np.int8) * x
        #
        # mviri_easy["Time"].data[:] = np.ones((13198), np.float64)
        mviri_easy["distance_sun_earth"].data = 3.3
        mviri_easy["solar_irradiance_vis"].data = 4.4
        mviri_easy["u_solar_irradiance_vis"].data = 5.5
        mviri_easy["spectral_response_function_vis"].data[:] = np.ones((1011), np.float32) * 0.03
        mviri_easy["spectral_response_function_ir"].data[:] = np.ones((1011), np.float32) * 0.04
        mviri_easy["u_spectral_response_function_ir"].data[:] = np.ones((1011), np.float32) * 0.05
        mviri_easy["spectral_response_function_wv"].data[:] = np.ones((1011), np.float32) * 0.06
        mviri_easy["u_spectral_response_function_wv"].data[:] = np.ones((1011), np.float32) * 0.07
        mviri_easy["a_ir"].data = 8.8
        mviri_easy["b_ir"].data = 9.9
        mviri_easy["u_a_ir"].data = 10.1
        mviri_easy["u_b_ir"].data = 11.11
        mviri_easy["a_wv"].data = 12.12
        mviri_easy["b_wv"].data = 13.13
        mviri_easy["u_a_wv"].data = 14.14
        mviri_easy["u_b_wv"].data = 15.15
        mviri_easy["q_ir"].data = 16.16
        mviri_easy["q_wv"].data = 17.17
        mviri_easy["unit_conversion_ir"].data = 18.18
        mviri_easy["unit_conversion_wv"].data = 19.19
        mviri_easy["bt_a_ir"].data = 20.2
        mviri_easy["bt_b_ir"].data = 21.21
        mviri_easy["bt_a_wv"].data = 22.22
        mviri_easy["bt_b_wv"].data = 23.23
        mviri_easy["years_since_launch"].data = 24.24
        mviri_easy["sub_satellite_latitude_start"].data = 25.25
        mviri_easy["sub_satellite_longitude_start"].data = 26.26
        mviri_easy["sub_satellite_latitude_end"].data = 27.27
        mviri_easy["sub_satellite_longitude_end"].data = 28.28

        return mviri_easy
