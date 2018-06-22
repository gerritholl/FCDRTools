import numpy as np


class Assertions:
    @staticmethod
    def assert_geolocation_variables(test_case, dataset, width, height, chunking=None):
        latitude = dataset.variables["latitude"]
        test_case.assertEqual((height, width), latitude.shape)
        test_case.assertTrue(np.isnan(latitude.data[1, 1]))
        test_case.assertEqual("latitude", latitude.attrs["standard_name"])
        test_case.assertEqual("degrees_north", latitude.attrs["units"])
        test_case.assertEqual(np.int16, latitude.encoding['dtype'])
        test_case.assertEqual(-32768, latitude.encoding['_FillValue'])
        test_case.assertEqual(0.0027466658, latitude.encoding['scale_factor'])
        test_case.assertEqual(0.0, latitude.encoding['add_offset'])
        if chunking is not None:
            test_case.assertEqual(chunking, latitude.encoding['chunksizes'])

        longitude = dataset.variables["longitude"]
        test_case.assertEqual((height, width), longitude.shape)
        test_case.assertTrue(np.isnan(longitude.data[2, 2]))
        test_case.assertEqual("longitude", longitude.attrs["standard_name"])
        test_case.assertEqual("degrees_east", longitude.attrs["units"])
        test_case.assertEqual(np.int16, longitude.encoding['dtype'])
        test_case.assertEqual(-32768, longitude.encoding['_FillValue'])
        test_case.assertEqual(0.0054933317, longitude.encoding['scale_factor'])
        test_case.assertEqual(0.0, longitude.encoding['add_offset'])
        if chunking is not None:
            test_case.assertEqual(chunking, longitude.encoding['chunksizes'])

    @staticmethod
    def assert_quality_flags(test_case, dataset, width, height, chunking=None, masks_append=None, meanings_append=None):
        quality = dataset.variables["quality_pixel_bitmask"]
        test_case.assertEqual((height, width), quality.shape)
        test_case.assertEqual(0, quality.data[1, 1])
        test_case.assertEqual("status_flag", quality.attrs["standard_name"])
        test_case.assertEqual("longitude latitude", quality.attrs["coordinates"])

        masks = "1, 2, 4, 8, 16, 32, 64, 128"
        if masks_append is not None:
            masks = masks + masks_append
        test_case.assertEqual(masks, quality.attrs["flag_masks"])

        meanings = "invalid use_with_caution invalid_input invalid_geoloc invalid_time sensor_error padded_data incomplete_channel_data"
        if meanings_append is not None:
            meanings = meanings + meanings_append
        test_case.assertEqual(meanings, quality.attrs["flag_meanings"])
        
        if chunking is not None:
            test_case.assertEqual(chunking, quality.encoding['chunksizes'])

    @staticmethod
    def assert_correlation_matrices(test_case, ds, num_channels):
        ch_corr_indep = ds.variables["channel_correlation_matrix_independent"]
        test_case.assertEqual((num_channels, num_channels), ch_corr_indep.shape)
        test_case.assertAlmostEqual(1.0, ch_corr_indep[0, 0])
        test_case.assertAlmostEqual(0.0, ch_corr_indep[1, 0])
        test_case.assertEqual(np.int16, ch_corr_indep.encoding['dtype'])
        test_case.assertEqual(-32768, ch_corr_indep.encoding['_FillValue'])
        test_case.assertEqual(0.0001, ch_corr_indep.encoding['scale_factor'])
        test_case.assertEqual(0.0, ch_corr_indep.encoding['add_offset'])
        test_case.assertEqual("Channel_correlation_matrix_independent_effects", ch_corr_indep.attrs["long_name"])
        test_case.assertEqual("1", ch_corr_indep.attrs["units"])
        test_case.assertEqual("-10000", ch_corr_indep.attrs["valid_min"])
        test_case.assertEqual("10000", ch_corr_indep.attrs["valid_max"])
        test_case.assertEqual("Channel error correlation matrix for independent effects", ch_corr_indep.attrs["description"])

        ch_corr_stuct = ds.variables["channel_correlation_matrix_structured"]
        test_case.assertEqual((num_channels, num_channels), ch_corr_stuct.shape)
        test_case.assertAlmostEqual(1.0, ch_corr_stuct[1, 1])
        test_case.assertAlmostEqual(0.0, ch_corr_stuct[0, 2])
        test_case.assertEqual(np.int16, ch_corr_indep.encoding['dtype'])
        test_case.assertEqual(-32768, ch_corr_indep.encoding['_FillValue'])
        test_case.assertEqual(0.0001, ch_corr_indep.encoding['scale_factor'])
        test_case.assertEqual(0.0, ch_corr_indep.encoding['add_offset'])
        test_case.assertEqual("Channel_correlation_matrix_structured_effects", ch_corr_stuct.attrs["long_name"])
        test_case.assertEqual("1", ch_corr_stuct.attrs["units"])
        test_case.assertEqual("-10000", ch_corr_indep.attrs["valid_min"])
        test_case.assertEqual("10000", ch_corr_indep.attrs["valid_max"])
        test_case.assertEqual("Channel error correlation matrix for structured effects", ch_corr_stuct.attrs["description"])
