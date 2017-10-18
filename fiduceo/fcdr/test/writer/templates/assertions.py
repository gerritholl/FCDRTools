import numpy as np


class Assertions:
    @staticmethod
    def assert_geolocation_variables(test_case, dataset, width, height):
        latitude = dataset.variables["latitude"]
        test_case.assertEqual((height, width), latitude.shape)
        test_case.assertTrue(np.isnan(latitude.data[1, 1]))
        test_case.assertEqual("latitude", latitude.attrs["standard_name"])
        test_case.assertEqual("degrees_north", latitude.attrs["units"])
        test_case.assertEqual(np.int16, latitude.encoding['dtype'])
        test_case.assertEqual(-32768, latitude.encoding['_FillValue'])
        test_case.assertEqual(0.0027466658, latitude.encoding['scale_factor'])
        test_case.assertEqual(0.0, latitude.encoding['add_offset'])

        longitude = dataset.variables["longitude"]
        test_case.assertEqual((height, width), longitude.shape)
        test_case.assertTrue(np.isnan(longitude.data[2, 2]))
        test_case.assertEqual("longitude", longitude.attrs["standard_name"])
        test_case.assertEqual("degrees_east", longitude.attrs["units"])
        test_case.assertEqual(np.int16, longitude.encoding['dtype'])
        test_case.assertEqual(-32768, longitude.encoding['_FillValue'])
        test_case.assertEqual(0.0054933317, longitude.encoding['scale_factor'])
        test_case.assertEqual(0.0, longitude.encoding['add_offset'])
