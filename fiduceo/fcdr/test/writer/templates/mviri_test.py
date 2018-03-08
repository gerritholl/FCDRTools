import unittest

import numpy as np
import xarray as xr

from fiduceo.fcdr.test.writer.templates.assertions import Assertions
from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.mviri import MVIRI

CHUNKING = (500, 500)


class MVIRITest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        MVIRI.add_original_variables(ds, 5)

        Assertions.assert_quality_flags(self, ds, 5000, 5000)

        time = ds.variables["time"]
        self.assertEqual((2500, 2500), time.shape)
        self.assertEqual(4294967295, time.data[4, 117])
        self.assertEqual(4294967295, time.attrs["_FillValue"])
        self.assertEqual("time", time.attrs["standard_name"])
        self.assertEqual("Acquisition time of pixel", time.attrs["long_name"])
        self.assertEqual("seconds since 1970-01-01 00:00:00", time.attrs["units"])
        self.assertEqual(CHUNKING, time.encoding["chunksizes"])
        self.assertEqual(-32768, time.attrs["add_offset"])

        sol_azimuth = ds.variables["solar_azimuth_angle"]
        self.assertEqual((5000, 5000), sol_azimuth.shape)
        self.assertTrue(np.isnan(sol_azimuth.data[0, 111]))
        self.assertEqual("solar_azimuth_angle", sol_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sol_azimuth.attrs["units"])
        self.assertEqual(np.uint16, sol_azimuth.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), sol_azimuth.encoding['_FillValue'])
        self.assertEqual(0.005493164, sol_azimuth.encoding['scale_factor'])
        self.assertEqual(0.0, sol_azimuth.encoding['add_offset'])
        self.assertEqual(CHUNKING, sol_azimuth.encoding["chunksizes"])

        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertEqual((5000, 5000), sol_zenith.shape)
        self.assertTrue(np.isnan(sol_zenith.data[0, 112]))
        self.assertEqual("solar_zenith_angle", sol_zenith.attrs["standard_name"])
        self.assertEqual("degree", sol_zenith.attrs["units"])
        self.assertEqual(np.int16, sol_zenith.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.encoding['_FillValue'])
        self.assertEqual(0.005493248, sol_zenith.encoding['scale_factor'])
        self.assertEqual(0.0, sol_zenith.encoding['add_offset'])
        self.assertEqual(CHUNKING, sol_zenith.encoding["chunksizes"])

        count = ds.variables["count_ir"]
        self.assertEqual((2500, 2500), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.data[0, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.attrs["_FillValue"])
        self.assertEqual("Infrared Image Counts", count.attrs["long_name"])
        self.assertEqual("count", count.attrs["units"])
        self.assertEqual(CHUNKING, count.encoding["chunksizes"])

        count = ds.variables["count_wv"]
        self.assertEqual((2500, 2500), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.data[1, 114])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.attrs["_FillValue"])
        self.assertEqual("WV Image Counts", count.attrs["long_name"])
        self.assertEqual("count", count.attrs["units"])
        self.assertEqual(CHUNKING, count.encoding["chunksizes"])

        dq_bitmask = ds.variables["data_quality_bitmask"]
        self.assertEqual((5000, 5000), dq_bitmask.shape)
        self.assertEqual(0, dq_bitmask.data[1, 6])
        self.assertEqual("1, 2, 4, 8, 16, 32", dq_bitmask.attrs["flag_masks"])
        self.assertEqual("uncertainty_suspicious uncertainty_too_large space_view_suspicious not_on_earth suspect_time suspect_geo", dq_bitmask.attrs["flag_meanings"])
        self.assertEqual("status_flag", dq_bitmask.attrs["standard_name"])

        self._assert_scalar_float_variable(ds, "distance_sun_earth", "Sun-Earth distance", "au")
        self._assert_scalar_float_variable(ds, "solar_irradiance_vis", "Solar effective Irradiance", "W*m-2", standard_name="solar_irradiance_vis")

        u_sol_irr = ds.variables["u_solar_irradiance_vis"]
        self.assertEqual((), u_sol_irr.shape)
        self.assertTrue(np.isnan(u_sol_irr.data))
        self.assertTrue(np.isnan(u_sol_irr.attrs["_FillValue"]))
        self.assertEqual("Uncertainty in Solar effective Irradiance", u_sol_irr.attrs["long_name"])
        self.assertEqual("Wm^-2", u_sol_irr.attrs["units"])
        self.assertEqual("rectangle_absolute", u_sol_irr.attrs["pixel_correlation_form"])
        self.assertEqual("pixel", u_sol_irr.attrs["pixel_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_sol_irr.attrs["pixel_correlation_scales"])
        self.assertEqual("rectangle_absolute", u_sol_irr.attrs["scan_correlation_form"])
        self.assertEqual("line", u_sol_irr.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_sol_irr.attrs["scan_correlation_scales"])
        self.assertEqual("rectangle_absolute", u_sol_irr.attrs["image_correlation_form"])
        self.assertEqual("days", u_sol_irr.attrs["image_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_sol_irr.attrs["image_correlation_scales"])
        self.assertEqual("rectangle", u_sol_irr.attrs["pdf_shape"])

        srf = ds.variables["spectral_response_function_vis"]
        self.assertEqual((1011,), srf.shape)
        self.assertTrue(np.isnan(srf.data[116]))
        self.assertTrue(np.isnan(srf.attrs["_FillValue"]))
        self.assertEqual("Spectral Response Function for visible channel", srf.attrs["long_name"])
        self.assertEqual("Filename of SRF", srf.attrs["source"])
        self.assertEqual("datestring", srf.attrs["Valid(YYYYDDD)"])

        cov_srf = ds.variables["covariance_spectral_response_function_vis"]
        self.assertEqual((1011, 1011), cov_srf.shape)
        self.assertTrue(np.isnan(cov_srf.data[116, 22]))
        self.assertTrue(np.isnan(cov_srf.attrs["_FillValue"]))
        self.assertEqual("Covariance of the Visible Band Spectral Response Function", cov_srf.attrs["long_name"])
        self.assertEqual(CHUNKING, cov_srf.encoding["chunksizes"])

        srf_ir = ds.variables["spectral_response_function_ir"]
        self.assertEqual((1011,), srf_ir.shape)
        self.assertTrue(np.isnan(srf_ir.data[118]))
        self.assertTrue(np.isnan(srf_ir.attrs["_FillValue"]))
        self.assertEqual("Spectral Response Function for IR channel", srf_ir.attrs["long_name"])

        u_srf_ir = ds.variables["u_spectral_response_function_ir"]
        self.assertEqual((1011,), u_srf_ir.shape)
        self.assertTrue(np.isnan(u_srf_ir.data[119]))
        self.assertTrue(np.isnan(u_srf_ir.attrs["_FillValue"]))
        self.assertEqual("Uncertainty in Spectral Response Function for IR channel", u_srf_ir.attrs["long_name"])

        srf_wv = ds.variables["spectral_response_function_wv"]
        self.assertEqual((1011,), srf_wv.shape)
        self.assertTrue(np.isnan(srf_wv.data[118]))
        self.assertTrue(np.isnan(srf_wv.attrs["_FillValue"]))
        self.assertEqual("Spectral Response Function for WV channel", srf_wv.attrs["long_name"])

        u_srf_wv = ds.variables["u_spectral_response_function_wv"]
        self.assertEqual((1011,), u_srf_wv.shape)
        self.assertTrue(np.isnan(u_srf_wv.data[120]))
        self.assertTrue(np.isnan(u_srf_wv.attrs["_FillValue"]))
        self.assertEqual("Uncertainty in Spectral Response Function for WV channel", u_srf_wv.attrs["long_name"])

        self._assert_scalar_float_variable(ds, "a_ir", "Calibration parameter a for IR Band", "mWm^-2sr^-1cm^-1")
        self._assert_scalar_float_variable(ds, "b_ir", "Calibration parameter b for IR Band", "mWm^-2sr^-1cm^-1/DC")
        self._assert_scalar_float_variable(ds, "u_a_ir", "Uncertainty of calibration parameter a for IR Band", "mWm^-2sr^-1cm^-1")
        self._assert_scalar_float_variable(ds, "u_b_ir", "Uncertainty of calibration parameter b for IR Band", "mWm^-2sr^-1cm^-1/DC")
        self._assert_scalar_float_variable(ds, "a_wv", "Calibration parameter a for WV Band", "mWm^-2sr^-1cm^-1")
        self._assert_scalar_float_variable(ds, "b_wv", "Calibration parameter b for WV Band", "mWm^-2sr^-1cm^-1/DC")
        self._assert_scalar_float_variable(ds, "u_a_wv", "Uncertainty of calibration parameter a for WV Band", "mWm^-2sr^-1cm^-1")
        self._assert_scalar_float_variable(ds, "u_b_wv", "Uncertainty of calibration parameter b for WV Band", "mWm^-2sr^-1cm^-1/DC")
        self._assert_scalar_float_variable(ds, "q_ir", "IR Band Calibration quality flag", "1")
        self._assert_scalar_float_variable(ds, "q_wv", "WV Band Calibration quality flag", "1")
        self._assert_scalar_float_variable(ds, "unit_conversion_ir", "IR Unit conversion factor", "1")
        self._assert_scalar_float_variable(ds, "unit_conversion_wv", "WV Unit conversion factor", "1")
        self._assert_scalar_float_variable(ds, "bt_a_ir", "IR Band BT conversion parameter A", "1")
        self._assert_scalar_float_variable(ds, "bt_b_ir", "IR Band BT conversion parameter B", "1")
        self._assert_scalar_float_variable(ds, "bt_a_wv", "WV Band BT conversion parameter A", "1")
        self._assert_scalar_float_variable(ds, "bt_b_wv", "WV Band BT conversion parameter B", "1")
        self._assert_scalar_float_variable(ds, "years_since_launch", "Fractional year since launch of satellite", "years")

        x = ds.coords["x"]
        self.assertEqual((5000,), x.shape)
        self.assertEqual(15, x[15])

        y = ds.coords["y"]
        self.assertEqual((5000,), y.shape)
        self.assertEqual(6, y[6])

        x_ir_wv = ds.coords["x_ir_wv"]
        self.assertEqual((2500,), x_ir_wv.shape)
        self.assertEqual(16, x_ir_wv[16])

        y_ir_wv = ds.coords["y_ir_wv"]
        self.assertEqual((2500,), y_ir_wv.shape)
        self.assertEqual(17, x[17])

        srf_size = ds.coords["srf_size"]
        self.assertEqual((1011,), srf_size.shape)
        self.assertEqual(18, srf_size[18])

    def test_get_swath_width(self):
        self.assertEqual(5000, MVIRI.get_swath_width())

    def test_add_easy_fcdr_variables(self):
        ds = xr.Dataset()
        MVIRI.add_easy_fcdr_variables(ds, 8)

        reflectance = ds.variables["toa_bidirectional_reflectance_vis"]
        self.assertEqual((5000, 5000), reflectance.shape)
        self.assertTrue(np.isnan(reflectance.data[3, 115]))
        self.assertEqual("toa_bidirectional_reflectance_vis", reflectance.attrs["standard_name"])
        self.assertEqual("top of atmosphere bidirectional reflectance factor per pixel of the visible band with central wavelength 0.7", reflectance.attrs["long_name"])
        self.assertEqual("percent", reflectance.attrs["units"])
        self.assertEqual(np.uint16, reflectance.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), reflectance.encoding['_FillValue'])
        self.assertEqual(1.52588E-05, reflectance.encoding['scale_factor'])
        self.assertEqual(0.0, reflectance.encoding['add_offset'])
        self.assertEqual(CHUNKING, reflectance.encoding["chunksizes"])

        u_indep = ds.variables["u_independent_toa_bidirectional_reflectance"]
        self.assertEqual((5000, 5000), u_indep.shape)
        self.assertTrue(np.isnan(u_indep.data[118, 234]))
        self.assertEqual("independent uncertainty per pixel", u_indep.attrs["long_name"])
        self.assertEqual("percent", u_indep.attrs["units"])
        self.assertEqual(np.uint16, u_indep.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_indep.encoding['_FillValue'])
        self.assertEqual(1.52588E-05, u_indep.encoding['scale_factor'])
        self.assertEqual(0.0, u_indep.encoding['add_offset'])
        self.assertEqual(CHUNKING, u_indep.encoding["chunksizes"])

        u_struct = ds.variables["u_structured_toa_bidirectional_reflectance"]
        self.assertEqual((5000, 5000), u_struct.shape)
        self.assertTrue(np.isnan(u_struct.data[119, 235]))
        self.assertEqual("structured uncertainty per pixel", u_struct.attrs["long_name"])
        self.assertEqual("percent", u_struct.attrs["units"])
        self.assertEqual(np.uint16, u_struct.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_struct.encoding['_FillValue'])
        self.assertEqual(1.52588E-05, u_struct.encoding['scale_factor'])
        self.assertEqual(0.0, u_struct.encoding['add_offset'])
        self.assertEqual(CHUNKING, u_struct.encoding["chunksizes"])

        self._assert_scalar_float_variable(ds, "sub_satellite_latitude_start", "Latitude of the sub satellite point at image start", "degrees_north")
        self._assert_scalar_float_variable(ds, "sub_satellite_longitude_start", "Longitude of the sub satellite point at image start", "degrees_east")
        self._assert_scalar_float_variable(ds, "sub_satellite_latitude_end", "Latitude of the sub satellite point at image end", "degrees_north")
        self._assert_scalar_float_variable(ds, "sub_satellite_longitude_end", "Longitude of the sub satellite point at image end", "degrees_east")

    def test_add_full_fcdr_variables(self):
        ds = xr.Dataset()
        MVIRI.add_full_fcdr_variables(ds, 7)

        count = ds.variables["count_vis"]
        self.assertEqual((5000, 5000), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.data[0, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.attrs["_FillValue"])
        self.assertEqual("Image counts", count.attrs["long_name"])
        self.assertEqual("count", count.attrs["units"])
        self.assertEqual(CHUNKING, count.encoding["chunksizes"])

        u_lat = ds.variables["u_latitude"]
        self.assertEqual((5000, 5000), u_lat.shape)
        self.assertTrue(np.isnan(u_lat.data[5, 109]))
        self.assertEqual("Uncertainty in Latitude", u_lat.attrs["long_name"])
        self.assertEqual("degree", u_lat.attrs["units"])
        self.assertEqual(np.uint16, u_lat.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_lat.encoding['_FillValue'])
        self.assertEqual(1.5E-05, u_lat.encoding['scale_factor'])
        self.assertEqual(0.0, u_lat.encoding['add_offset'])
        self.assertEqual("triangle_relative", u_lat.attrs["pixel_correlation_form"])
        self.assertEqual("pixel", u_lat.attrs["pixel_correlation_units"])
        self.assertEqual([-250, 250], u_lat.attrs["pixel_correlation_scales"])
        self.assertEqual("triangle_relative", u_lat.attrs["scan_correlation_form"])
        self.assertEqual("line", u_lat.attrs["scan_correlation_units"])
        self.assertEqual([-250, 250], u_lat.attrs["scan_correlation_scales"])
        self.assertEqual("triangle_relative", u_lat.attrs["image_correlation_form"])
        self.assertEqual("images", u_lat.attrs["image_correlation_units"])
        self.assertEqual([-12, 0], u_lat.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_lat.attrs["pdf_shape"])
        self.assertEqual(CHUNKING, u_lat.encoding["chunksizes"])

        u_lon = ds.variables["u_longitude"]
        self.assertEqual((5000, 5000), u_lon.shape)
        self.assertTrue(np.isnan(u_lon.data[6, 110]))
        self.assertEqual("Uncertainty in Longitude", u_lon.attrs["long_name"])
        self.assertEqual("degree", u_lon.attrs["units"])
        self.assertEqual(np.uint16, u_lon.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_lon.encoding['_FillValue'])
        self.assertEqual(1.5E-05, u_lon.encoding['scale_factor'])
        self.assertEqual(0.0, u_lon.encoding['add_offset'])
        self.assertEqual("triangle_relative", u_lon.attrs["pixel_correlation_form"])
        self.assertEqual("pixel", u_lon.attrs["pixel_correlation_units"])
        self.assertEqual([-250, 250], u_lon.attrs["pixel_correlation_scales"])
        self.assertEqual("triangle_relative", u_lon.attrs["scan_correlation_form"])
        self.assertEqual("line", u_lon.attrs["scan_correlation_units"])
        self.assertEqual([-250, 250], u_lon.attrs["scan_correlation_scales"])
        self.assertEqual("triangle_relative", u_lon.attrs["image_correlation_form"])
        self.assertEqual("images", u_lon.attrs["image_correlation_units"])
        self.assertEqual([-12, 0], u_lon.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_lon.attrs["pdf_shape"])
        self.assertEqual(CHUNKING, u_lon.encoding["chunksizes"])

        u_time = ds.variables["u_time"]
        self.assertEqual((2500,), u_time.shape)
        self.assertTrue(np.isnan(u_time.data[111]))
        self.assertEqual("Uncertainty in Time", u_time.attrs["standard_name"])
        self.assertEqual("s", u_time.attrs["units"])
        self.assertEqual(np.uint16, u_time.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_time.encoding['_FillValue'])
        self.assertEqual(0.009155273, u_time.encoding['scale_factor'])
        self.assertEqual(0.0, u_time.encoding['add_offset'])
        self.assertEqual("rectangle", u_time.attrs["pdf_shape"])

        u_sat_zenith = ds.variables["u_satellite_zenith_angle"]
        self.assertEqual((5000, 5000), u_sat_zenith.shape)
        self.assertTrue(np.isnan(u_sat_zenith.data[1, 112]))
        self.assertEqual("Uncertainty in Satellite Zenith Angle", u_sat_zenith.attrs["long_name"])
        self.assertEqual("degree", u_sat_zenith.attrs["units"])
        self.assertEqual(np.uint16, u_sat_zenith.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sat_zenith.encoding['_FillValue'])
        self.assertEqual(7.62939E-05, u_sat_zenith.encoding['scale_factor'])
        self.assertEqual(0.0, u_sat_zenith.encoding['add_offset'])
        self.assertEqual(CHUNKING, u_sat_zenith.encoding["chunksizes"])

        u_sat_azimuth = ds.variables["u_satellite_azimuth_angle"]
        self.assertEqual((5000, 5000), u_sat_azimuth.shape)
        self.assertTrue(np.isnan(u_sat_azimuth.data[2, 113]))
        self.assertEqual("Uncertainty in Satellite Azimuth Angle", u_sat_azimuth.attrs["long_name"])
        self.assertEqual("degree", u_sat_azimuth.attrs["units"])
        self.assertEqual(np.uint16, u_sat_azimuth.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sat_azimuth.encoding['_FillValue'])
        self.assertEqual(7.62939E-05, u_sat_azimuth.encoding['scale_factor'])
        self.assertEqual(0.0, u_sat_azimuth.encoding['add_offset'])
        self.assertEqual(CHUNKING, u_sat_azimuth.encoding["chunksizes"])

        u_sol_zenith = ds.variables["u_solar_zenith_angle"]
        self.assertEqual((5000, 5000), u_sol_zenith.shape)
        self.assertTrue(np.isnan(u_sol_zenith.data[3, 114]))
        self.assertEqual("Uncertainty in Solar Zenith Angle", u_sol_zenith.attrs["long_name"])
        self.assertEqual("degree", u_sol_zenith.attrs["units"])
        self.assertEqual(np.uint16, u_sol_zenith.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sol_zenith.encoding['_FillValue'])
        self.assertEqual(7.62939E-05, u_sol_zenith.encoding['scale_factor'])
        self.assertEqual(0.0, u_sol_zenith.encoding['add_offset'])
        self.assertEqual(CHUNKING, u_sol_zenith.encoding["chunksizes"])

        u_sol_azimuth = ds.variables["u_solar_azimuth_angle"]
        self.assertEqual((5000, 5000), u_sol_azimuth.shape)
        self.assertTrue(np.isnan(u_sol_azimuth.data[4, 115]))
        self.assertEqual("Uncertainty in Solar Azimuth Angle", u_sol_azimuth.attrs["long_name"])
        self.assertEqual("degree", u_sol_azimuth.attrs["units"])
        self.assertEqual(np.uint16, u_sol_azimuth.encoding['dtype'])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sol_azimuth.encoding['_FillValue'])
        self.assertEqual(7.62939E-05, u_sol_azimuth.encoding['scale_factor'])
        self.assertEqual(0.0, u_sol_azimuth.encoding['add_offset'])
        self.assertEqual(CHUNKING, u_sol_azimuth.encoding["chunksizes"])

        a0 = ds.variables["a0_vis"]
        self.assertEqual((), a0.shape)
        self.assertTrue(np.isnan(a0.data))
        self.assertTrue(np.isnan(a0.attrs["_FillValue"]))
        self.assertEqual("Calibration Coefficient at Launch", a0.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count", a0.attrs["units"])

        a1 = ds.variables["a1_vis"]
        self.assertEqual((), a1.shape)
        self.assertTrue(np.isnan(a1.data))
        self.assertTrue(np.isnan(a1.attrs["_FillValue"]))
        self.assertEqual("Time variation of a0", a1.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count day^-1 10^5", a1.attrs["units"])

        k_space = ds.variables["mean_count_space_vis"]
        self.assertEqual((), k_space.shape)
        self.assertTrue(np.isnan(k_space.data))
        self.assertTrue(np.isnan(k_space.attrs["_FillValue"]))
        self.assertEqual("Space count", k_space.attrs["long_name"])
        self.assertEqual("count", k_space.attrs["units"])

        u_a0_vis = ds.variables["u_a0_vis"]
        self.assertEqual((), u_a0_vis.shape)
        self.assertTrue(np.isnan(u_a0_vis.data))
        self.assertTrue(np.isnan(u_a0_vis.attrs["_FillValue"]))
        self.assertEqual("Uncertainty in a0", u_a0_vis.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count", u_a0_vis.attrs["units"])
        self.assertEqual("rectangle_absolute", u_a0_vis.attrs["pixel_correlation_form"])
        self.assertEqual("pixel", u_a0_vis.attrs["pixel_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_a0_vis.attrs["pixel_correlation_scales"])
        self.assertEqual("rectangle_absolute", u_a0_vis.attrs["scan_correlation_form"])
        self.assertEqual("line", u_a0_vis.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_a0_vis.attrs["scan_correlation_scales"])
        self.assertEqual("triangle_relative", u_a0_vis.attrs["image_correlation_form"])
        self.assertEqual("months", u_a0_vis.attrs["image_correlation_units"])
        self.assertEqual([-1.5, 1.5], u_a0_vis.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_a0_vis.attrs["pdf_shape"])

        u_a1_vis = ds.variables["u_a1_vis"]
        self.assertEqual((), u_a0_vis.shape)
        self.assertTrue(np.isnan(u_a1_vis.data))
        self.assertTrue(np.isnan(u_a1_vis.attrs["_FillValue"]))
        self.assertEqual("Uncertainty in a1", u_a1_vis.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count day^-1 10^5", u_a1_vis.attrs["units"])
        self.assertEqual("rectangle_absolute", u_a1_vis.attrs["pixel_correlation_form"])
        self.assertEqual("pixel", u_a1_vis.attrs["pixel_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_a1_vis.attrs["pixel_correlation_scales"])
        self.assertEqual("rectangle_absolute", u_a1_vis.attrs["scan_correlation_form"])
        self.assertEqual("line", u_a1_vis.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_a1_vis.attrs["scan_correlation_scales"])
        self.assertEqual("triangle_relative", u_a1_vis.attrs["image_correlation_form"])
        self.assertEqual("months", u_a1_vis.attrs["image_correlation_units"])
        self.assertEqual([-1.5, 1.5], u_a1_vis.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_a1_vis.attrs["pdf_shape"])

        u_a0_a1_cov = ds.variables["covariance_a0_a1_vis"]
        self.assertEqual((), u_a0_a1_cov.shape)
        self.assertTrue(np.isnan(u_a0_a1_cov.data))
        self.assertTrue(np.isnan(u_a0_a1_cov.attrs["_FillValue"]))
        self.assertEqual("Covariance of calibration coefficients", u_a0_a1_cov.attrs["long_name"])

        u_e_noise = ds.variables["u_electronics_counts_vis"]
        self.assertEqual((), u_e_noise.shape)
        self.assertTrue(np.isnan(u_e_noise.data))
        self.assertTrue(np.isnan(u_e_noise.attrs["_FillValue"]))
        self.assertEqual("Uncertainty due to Electronics noise", u_e_noise.attrs["long_name"])
        self.assertEqual("count", u_e_noise.attrs["units"])

        u_digitization = ds.variables["u_digitization_counts_vis"]
        self.assertEqual((), u_digitization.shape)
        self.assertTrue(np.isnan(u_digitization.data))
        self.assertTrue(np.isnan(u_digitization.attrs["_FillValue"]))
        self.assertEqual("Uncertainty due to digitization", u_digitization.attrs["long_name"])
        self.assertEqual("count", u_digitization.attrs["units"])

        u_space = ds.variables["allan_deviation_counts_space_vis"]
        self.assertEqual((), u_space.shape)
        self.assertTrue(np.isnan(u_space.data))
        self.assertTrue(np.isnan(u_space.attrs["_FillValue"]))
        self.assertEqual("Uncertainty of space count", u_space.attrs["long_name"])
        self.assertEqual("count", u_space.attrs["units"])
        self.assertEqual("rectangle_absolute", u_space.attrs["scan_correlation_form"])
        self.assertEqual("line", u_space.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_space.attrs["scan_correlation_scales"])
        self.assertEqual("digitised_gaussian", u_space.attrs["pdf_shape"])

        s_sol_irr_vis = ds.variables["sensitivity_solar_irradiance_vis"]
        self.assertEqual((), s_sol_irr_vis.shape)
        self.assertTrue(np.isnan(s_sol_irr_vis.data))
        self.assertEqual("true", s_sol_irr_vis.attrs["virtual"])
        self.assertEqual("y, x", s_sol_irr_vis.attrs["dimension"])
        self.assertEqual("distance_sun_earth * distance_sun_earth * PI * (count_vis - mean_count_space_vis) * (a1_vis * years_since_launch + a0_vis) / (cos(solar_zenith_angle * PI / 180.0) * solar_irradiance_vis * solar_irradiance_vis)", s_sol_irr_vis.attrs["expression"])

        s_count_vis = ds.variables["sensitivity_count_vis"]
        self.assertEqual((), s_count_vis.shape)
        self.assertTrue(np.isnan(s_count_vis.data))
        self.assertEqual("true", s_count_vis.attrs["virtual"])
        self.assertEqual("y, x", s_count_vis.attrs["dimension"])
        self.assertEqual("distance_sun_earth * distance_sun_earth * PI * (a1_vis * years_since_launch + a0_vis) / (cos(solar_zenith_angle * PI / 180.0) * solar_irradiance_vis)", s_count_vis.attrs["expression"])

        s_count_space = ds.variables["sensitivity_count_space"]
        self.assertEqual((), s_count_space.shape)
        self.assertTrue(np.isnan(s_count_space.data))
        self.assertEqual("true", s_count_space.attrs["virtual"])
        self.assertEqual("y, x", s_count_space.attrs["dimension"])
        self.assertEqual("-1.0 * distance_sun_earth * distance_sun_earth * PI * (a1_vis * years_since_launch + a0_vis) / (cos(solar_zenith_angle * PI / 180.0) * solar_irradiance_vis)", s_count_space.attrs["expression"])

        s_a0_vis = ds.variables["sensitivity_a0_vis"]
        self.assertEqual((), s_a0_vis.shape)
        self.assertTrue(np.isnan(s_a0_vis.data))
        self.assertEqual("true", s_a0_vis.attrs["virtual"])
        self.assertEqual("y, x", s_a0_vis.attrs["dimension"])
        self.assertEqual("distance_sun_earth * distance_sun_earth * PI * (count_vis - mean_count_space_vis) / (cos(solar_zenith_angle * PI / 180.0) * solar_irradiance_vis)", s_a0_vis.attrs["expression"])

        s_a1_vis = ds.variables["sensitivity_a1_vis"]
        self.assertEqual((), s_a1_vis.shape)
        self.assertTrue(np.isnan(s_a1_vis.data))
        self.assertEqual("true", s_a1_vis.attrs["virtual"])
        self.assertEqual("y, x", s_a1_vis.attrs["dimension"])
        self.assertEqual("distance_sun_earth * distance_sun_earth * PI * (count_vis - mean_count_space_vis) * years_since_launch / (cos(solar_zenith_angle * PI / 180.0) * solar_irradiance_vis)", s_a1_vis.attrs["expression"])

    def test_add_template_key(self):
        ds = xr.Dataset()

        MVIRI.add_template_key(ds)

        self.assertEqual("MVIRI", ds.attrs["template_key"])

    def _assert_scalar_float_variable(self, ds, name, long_name, units, standard_name=None):
        dse = ds.variables[name]
        self.assertEqual((), dse.shape)
        self.assertTrue(np.isnan(dse.data))
        self.assertTrue(np.isnan(dse.attrs["_FillValue"]))

        if standard_name is not None:
            self.assertEqual(standard_name, dse.attrs["standard_name"])

        self.assertEqual(long_name, dse.attrs["long_name"])
        self.assertEqual(units, dse.attrs["units"])
