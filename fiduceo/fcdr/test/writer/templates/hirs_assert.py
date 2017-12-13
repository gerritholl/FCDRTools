import unittest

import numpy as np

from fiduceo.fcdr.test.writer.templates.assertions import Assertions
from fiduceo.fcdr.writer.default_data import DefaultData

CHUNKING_2D = (512, 56)

class HIRSAssert(unittest.TestCase):
    # this is a mean hack - there is some error in the framework when using Python 2.7 that requests this method tb 2017-05-10
    def runTest(self):
        pass

    def assert_bt_variable(self, ds, chunking=None):
        bt = ds.variables["bt"]
        self.assertEqual((19, 6, 56), bt.shape)
        self.assertTrue(np.isnan(bt.data[0, 2, 1]))
        self.assertEqual("toa_brightness_temperature", bt.attrs["standard_name"])
        self.assertEqual("Brightness temperature, NOAA/EUMETSAT calibrated", bt.attrs["long_name"])
        self.assertEqual("K", bt.attrs["units"])
        self.assertEqual(np.int16, bt.encoding['dtype'])
        self.assertEqual(-999, bt.encoding['_FillValue'])
        self.assertEqual(0.01, bt.encoding['scale_factor'])
        self.assertEqual(150.0, bt.encoding['add_offset'])
        if chunking is not None:
            self.assertEqual(chunking, bt.encoding['chunksizes'])

        self.assertEqual("quality_scanline_bitmask quality_channel_bitmask", bt.attrs["ancilliary_variables"])

    def assert_common_angles(self, ds, chunking=None):
        satellite_zenith_angle = ds.variables["satellite_zenith_angle"]
        self.assertEqual((6, 56), satellite_zenith_angle.shape)
        self.assertTrue(np.isnan(satellite_zenith_angle.data[2, 2]))
        self.assertEqual(np.uint16, satellite_zenith_angle.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), satellite_zenith_angle.encoding['_FillValue'])
        self.assertEqual(0.01, satellite_zenith_angle.encoding['scale_factor'])
        self.assertEqual(-180.0, satellite_zenith_angle.encoding['add_offset'])
        if chunking is not None:
            self.assertEqual(chunking, satellite_zenith_angle.encoding['chunksizes'])
        self.assertEqual("platform_zenith_angle", satellite_zenith_angle.attrs["standard_name"])
        self.assertEqual("degree", satellite_zenith_angle.attrs["units"])

        solar_zenith_angle = ds.variables["solar_zenith_angle"]
        self.assertEqual((6, 56), solar_zenith_angle.shape)
        self.assertTrue(np.isnan(solar_zenith_angle.data[3, 3]))
        self.assertEqual(np.uint16, solar_zenith_angle.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), solar_zenith_angle.encoding['_FillValue'])
        self.assertEqual(0.01, solar_zenith_angle.encoding['scale_factor'])
        self.assertEqual(-180.0, solar_zenith_angle.encoding['add_offset'])
        if chunking is not None:
            self.assertEqual(chunking, solar_zenith_angle.encoding['chunksizes'])
        self.assertEqual("solar_zenith_angle", solar_zenith_angle.attrs["standard_name"])
        self.assertEqual("solar_zenith_angle", solar_zenith_angle.attrs["orig_name"])
        self.assertEqual("degree", solar_zenith_angle.attrs["units"])

        satellite_azimuth_angle = ds.variables["satellite_azimuth_angle"]
        self.assertEqual((6, 56), satellite_azimuth_angle.shape)
        self.assertTrue(np.isnan(satellite_azimuth_angle.data[5, 5]))
        self.assertEqual(np.uint16, satellite_azimuth_angle.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), satellite_azimuth_angle.encoding['_FillValue'])
        self.assertEqual(0.01, satellite_azimuth_angle.encoding['scale_factor'])
        self.assertEqual(-180.0, satellite_azimuth_angle.encoding['add_offset'])
        if chunking is not None:
            self.assertEqual(chunking, satellite_azimuth_angle.encoding['chunksizes'])
        self.assertEqual("sensor_azimuth_angle", satellite_azimuth_angle.attrs["standard_name"])
        self.assertEqual("local_azimuth_angle", satellite_azimuth_angle.attrs["long_name"])
        self.assertEqual("degree", satellite_azimuth_angle.attrs["units"])

        solar_azimuth_angle = ds.variables["solar_azimuth_angle"]
        self.assertEqual((6, 56), solar_azimuth_angle.shape)
        self.assertTrue(np.isnan(solar_azimuth_angle.data[4, 4]))
        self.assertEqual(np.uint16, solar_azimuth_angle.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), solar_azimuth_angle.encoding['_FillValue'])
        self.assertEqual(0.01, solar_azimuth_angle.encoding['scale_factor'])
        self.assertEqual(-180.0, solar_azimuth_angle.encoding['add_offset'])
        if chunking is not None:
            self.assertEqual(chunking, solar_azimuth_angle.encoding['chunksizes'])
        self.assertEqual("solar_azimuth_angle", solar_azimuth_angle.attrs["standard_name"])
        self.assertEqual("degree", solar_azimuth_angle.attrs["units"])

    def assert_common_sensor_variables(self, ds):
        scanline = ds.variables["scanline"]
        self.assertEqual((6,), scanline.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.data[3])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.attrs["_FillValue"])
        self.assertEqual("scanline_number", scanline.attrs["long_name"])
        self.assertEqual("count", scanline.attrs["units"])
        time = ds.variables["time"]
        self.assertEqual((6,), time.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint32), time.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint32), time.attrs["_FillValue"])
        self.assertEqual("time", time.attrs["standard_name"])
        self.assertEqual("Acquisition time in seconds since 1970-01-01 00:00:00", time.attrs["long_name"])
        self.assertEqual("s", time.attrs["units"])
        qual_scan_bitmask = ds.variables["quality_scanline_bitmask"]
        self.assertEqual((6,), qual_scan_bitmask.shape)
        self.assertEqual(0, qual_scan_bitmask.data[5])
        self.assertEqual("1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456",
                         qual_scan_bitmask.attrs["flag_masks"])
        self.assertEqual(
            "do_not_use_scan time_sequence_error data_gap_preceding_scan no_calibration no_earth_location clock_update status_changed line_incomplete, time_field_bad time_field_bad_not_inf inconsistent_sequence scan_time_repeat uncalib_bad_time calib_few_scans uncalib_bad_prt calib_marginal_prt uncalib_channels uncalib_inst_mode quest_ant_black_body zero_loc bad_loc_time bad_loc_marginal bad_loc_reason bad_loc_ant",
            qual_scan_bitmask.attrs["flag_meanings"])
        self.assertEqual("status_flag", qual_scan_bitmask.attrs["standard_name"])
        self.assertEqual("quality_indicator_bitfield", qual_scan_bitmask.attrs["long_name"])

    def assert_extended_quality_flags(self, ds):
        chqualflags = ds.variables["quality_channel_bitmask"]
        self.assertEqual((6, 19), chqualflags.shape)
        self.assertEqual(0, chqualflags.data[1, 2])
        self.assertEqual("status_flag", chqualflags.attrs["standard_name"])
        self.assertEqual("channel_quality_flags_bitfield", chqualflags.attrs["long_name"])

    def assert_minor_frame_flags(self, ds):
        mnfrqualflags = ds.variables["mnfrqualflags"]
        self.assertEqual((6, 64), mnfrqualflags.shape)
        self.assertEqual(0, mnfrqualflags.data[2, 5])
        self.assertEqual("status_flag", mnfrqualflags.attrs["standard_name"])
        self.assertEqual("minor_frame_quality_flags_bitfield", mnfrqualflags.attrs["long_name"])

    def assert_easy_fcdr_uncertainties(self, ds, chunking=None):
        self._assert_3d_channel_variable(ds, "u_independent", "uncertainty from independent errors", chunking=chunking)
        self._assert_3d_channel_variable(ds, "u_structured", "uncertainty from structured errors", chunking=chunking)

    def assert_global_flags(self, ds):
        hirs_masks = ", 128, 256, 512, 1024, 2048, 4096, 8192, 16384"
        hirs_meanings = " uncertainty_suspicious self_emission_fails calibration_impossible suspect_calib suspect_mirror_any reduced_context uncertainty_suspicious bad_temp_no_rself"
        Assertions.assert_quality_flags(self, ds, 56, 6, chunking=CHUNKING_2D, masks_append=hirs_masks, meanings_append=hirs_meanings)

    def _assert_3d_channel_variable(self, ds, name, long_name, chunking=None):
        variable = ds.variables[name]
        self.assertEqual((19, 7, 56), variable.shape)
        self.assertTrue(np.isnan(variable.data[2, 5, 3]))
        self.assertEqual(65535, variable.encoding["_FillValue"])
        self.assertEqual(0.001, variable.encoding["scale_factor"])
        self.assertEqual(np.uint16, variable.encoding['dtype'])
        if chunking is not None:
            self.assertEqual(chunking, variable.encoding['chunksizes'])
            
        self.assertEqual(long_name, variable.attrs["long_name"])
        self.assertEqual("K", variable.attrs["units"])
        self.assertEqual(1, variable.attrs["valid_min"])
        self.assertEqual(65534, variable.attrs["valid_max"])
