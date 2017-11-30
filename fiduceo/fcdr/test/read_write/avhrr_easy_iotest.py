import datetime
import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from fiduceo.fcdr.writer.fcdr_writer import FCDRWriter


class AvhrrEASYIoTest(unittest.TestCase):
    temp_dir = None
    target_path = None

    def setUp(self):
        self.temp_dir = tempfile.gettempdir()
        self.target_path = None

    def tearDown(self):
        if self.target_path is not None:
            os.remove(self.target_path)

    def test_write(self):
        avhrr_easy = self.create_easy_dataset()

        start = datetime.datetime(2016, 11, 22, 13, 24, 52)
        end = datetime.datetime(2016, 11, 22, 14, 25, 53)
        file_name = FCDRWriter.create_file_name_FCDR_easy("AVHRR", "NOAA18", start, end, "1.0")
        self.target_path = os.path.join(self.temp_dir, file_name)

        start_time = datetime.datetime.now()

        FCDRWriter.write(avhrr_easy, self.target_path)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print("AVHRR EASY write time: " + str(elapsed_time.seconds) + "." + str(round(elapsed_time.microseconds / 1000)))

        self.assertTrue(os.path.isfile(self.target_path))

        # target_data = xr.open_dataset(self.target_path, chunks=128)
        target_data = xr.open_dataset(self.target_path)
        try:
            variable = target_data["Ch1_Ref"]
            self.assertAlmostEqual(0.0, variable.data[0, 0], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["Ch2_Ref"]
            self.assertAlmostEqual(0.002, variable.data[1, 1], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["Ch3a_Ref"]
            self.assertAlmostEqual(0.006, variable.data[2, 2], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["Ch4_Bt"]
            self.assertAlmostEqual(0.01, variable.data[3, 3], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["Ch5_Bt"]
            self.assertAlmostEqual(0.02, variable.data[4, 4], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["Time"]
            self.assertAlmostEqual(1.0, variable.data[5], 8)
            self.assertEqual((13198,), variable.encoding["chunksizes"])

            variable = target_data["latitude"]
            self.assertAlmostEqual(0.041199987, variable.data[6, 6], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["longitude"]
            self.assertAlmostEqual(0.0494399853, variable.data[7, 7], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["quality_channel_bitmask"]
            self.assertEqual(1, variable.data[8, 1])
            self.assertEqual((13198, 6), variable.encoding["chunksizes"])

            variable = target_data["quality_pixel_bitmask"]
            self.assertEqual(9, variable.data[9, 9])
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["quality_scanline_bitmask"]
            self.assertEqual(1, variable.data[10])
            self.assertEqual((13198,), variable.encoding["chunksizes"])

            variable = target_data["relative_azimuth_angle"]
            self.assertAlmostEqual(0.11, variable.data[11, 11], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["satellite_zenith_angle"]
            self.assertAlmostEqual(0.13, variable.data[12, 12], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["solar_zenith_angle"]
            self.assertAlmostEqual(0.16, variable.data[13, 13], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_independent_Ch1"]
            self.assertAlmostEqual(0.182, variable.data[14, 14], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_independent_Ch2"]
            self.assertAlmostEqual(0.21, variable.data[15, 15], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_independent_Ch3a"]
            self.assertAlmostEqual(0.24, variable.data[16, 16], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_independent_Ch4"]
            self.assertAlmostEqual(0.272, variable.data[17, 17], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_independent_Ch5"]
            self.assertAlmostEqual(0.306, variable.data[18, 18], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_structured_Ch1"]
            self.assertAlmostEqual(0.34, variable.data[19, 19], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_structured_Ch2"]
            self.assertAlmostEqual(0.38, variable.data[20, 20], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_structured_Ch3a"]
            self.assertAlmostEqual(0.42, variable.data[21, 21], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_structured_Ch4"]
            self.assertAlmostEqual(0.462, variable.data[22, 22], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])

            variable = target_data["u_structured_Ch5"]
            self.assertAlmostEqual(0.506, variable.data[23, 23], 8)
            self.assertEqual((1024, 409), variable.encoding["chunksizes"])
        finally:
            target_data.close()

    def create_easy_dataset(self):
        avhrr_easy = FCDRWriter.createTemplateEasy("AVHRR", 13198)
        avhrr_easy.attrs["institution"] = "test"
        avhrr_easy.attrs["title"] = "sir"
        avhrr_easy.attrs["source"] = "invention"
        avhrr_easy.attrs["history"] = "new"
        avhrr_easy.attrs["references"] = "myself"
        avhrr_easy.attrs["comment"] = "should define a test version of this set"

        for x in range(0, 409):
            avhrr_easy["Ch1_Ref"].data[:, x] = np.ones((13198), np.int16) * x * 0.001
            avhrr_easy["Ch2_Ref"].data[:, x] = np.ones((13198), np.int16) * x * 0.002
            avhrr_easy["Ch3a_Ref"].data[:, x] = np.ones((13198), np.int16) * x * 0.003
            avhrr_easy["Ch4_Bt"].data[:, x] = np.ones((13198), np.int16) * x * 0.004
            avhrr_easy["Ch5_Bt"].data[:, x] = np.ones((13198), np.int16) * x * 0.005
            avhrr_easy["latitude"].data[:, x] = np.ones((13198), np.int16) * x * 0.006
            avhrr_easy["longitude"].data[:, x] = np.ones((13198), np.int16) * x * 0.007
            avhrr_easy["quality_pixel_bitmask"].data[:, x] = np.ones((13198), np.int8) * x
            avhrr_easy["relative_azimuth_angle"].data[:, x] = np.ones((13198), np.int16) * x * 0.010
            avhrr_easy["satellite_zenith_angle"].data[:, x] = np.ones((13198), np.int16) * x * 0.011
            avhrr_easy["solar_zenith_angle"].data[:, x] = np.ones((13198), np.int16) * x * 0.012
            avhrr_easy["u_independent_Ch1"].data[:, x] = np.ones((13198), np.int16) * x * 0.013
            avhrr_easy["u_independent_Ch2"].data[:, x] = np.ones((13198), np.int16) * x * 0.014
            avhrr_easy["u_independent_Ch3a"].data[:, x] = np.ones((13198), np.int16) * x * 0.015
            avhrr_easy["u_independent_Ch4"].data[:, x] = np.ones((13198), np.int16) * x * 0.016
            avhrr_easy["u_independent_Ch5"].data[:, x] = np.ones((13198), np.int16) * x * 0.017
            avhrr_easy["u_structured_Ch1"].data[:, x] = np.ones((13198), np.int16) * x * 0.018
            avhrr_easy["u_structured_Ch2"].data[:, x] = np.ones((13198), np.int16) * x * 0.019
            avhrr_easy["u_structured_Ch3a"].data[:, x] = np.ones((13198), np.int16) * x * 0.020
            avhrr_easy["u_structured_Ch4"].data[:, x] = np.ones((13198), np.int16) * x * 0.021
            avhrr_easy["u_structured_Ch5"].data[:, x] = np.ones((13198), np.int16) * x * 0.022

        for x in range(0, 6):
            avhrr_easy["quality_channel_bitmask"].data[:, x] = np.ones((13198), np.int8) * x

        avhrr_easy["Time"].data[:] = np.ones((13198), np.float64)
        avhrr_easy["quality_scanline_bitmask"].data[:] = np.ones((13198), np.int8)

        return avhrr_easy
