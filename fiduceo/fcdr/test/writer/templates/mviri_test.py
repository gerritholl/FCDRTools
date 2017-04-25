import unittest

import numpy as np
import xarray as xr
from fiduceo.fcdr.writer.default_data import DefaultData

from fiduceo.fcdr.writer.templates.mviri import MVIRI


class MVIRITest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        MVIRI.add_original_variables(ds, 5)

        time = ds.variables["time"]
        self.assertEqual((2500, 2500), time.shape)
        self.assertEqual(65535, time.data[4, 117])
        self.assertEqual(65535, time.attrs["_FillValue"])
        self.assertEqual("time", time.attrs["standard_name"])
        self.assertEqual("Acquisition time of pixel", time.attrs["long_name"])
        self.assertEqual("seconds since 1970-01-01 00:00:00", time.attrs["units"])
        self.assertEqual(-32768, time.attrs["add_offset"])

        sat_azimuth = ds.variables["satellite_azimuth_angle"]
        self.assertEqual((5000, 5000), sat_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), sat_azimuth.data[0, 109])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), sat_azimuth.attrs["_FillValue"])
        self.assertEqual("sensor_azimuth_angle", sat_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sat_azimuth.attrs["units"])
        self.assertEqual(0.005493164, sat_azimuth.attrs["scale_factor"])

        sat_zenith = ds.variables["satellite_zenith_angle"]
        self.assertEqual((5000, 5000), sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.data[0, 110])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_zenith.attrs["standard_name"])
        self.assertEqual("degree", sat_zenith.attrs["units"])
        self.assertEqual(0.005493248, sat_zenith.attrs["scale_factor"])

        sol_azimuth = ds.variables["solar_azimuth_angle"]
        self.assertEqual((5000, 5000), sol_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), sol_azimuth.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), sol_azimuth.attrs["_FillValue"])
        self.assertEqual("solar_azimuth_angle", sol_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sol_azimuth.attrs["units"])
        self.assertEqual(0.005493164, sol_azimuth.attrs["scale_factor"])

        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertEqual((5000, 5000), sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.data[0, 112])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.attrs["_FillValue"])
        self.assertEqual("solar_zenith_angle", sol_zenith.attrs["standard_name"])
        self.assertEqual("degree", sol_zenith.attrs["units"])
        self.assertEqual(0.005493248, sol_zenith.attrs["scale_factor"])

        count = ds.variables["count_ir"]
        self.assertEqual((2500, 2500), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.data[0, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.attrs["_FillValue"])
        self.assertEqual("Infrared Image Counts", count.attrs["long_name"])
        self.assertEqual("count", count.attrs["units"])

        count = ds.variables["count_wv"]
        self.assertEqual((2500, 2500), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.data[1, 114])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.attrs["_FillValue"])
        self.assertEqual("WV Image Counts", count.attrs["long_name"])
        self.assertEqual("count", count.attrs["units"])

        self._assert_scalar_float_variable(ds, "distance_sun_earth", "Sun-Earth distance", "au")
        self._assert_scalar_float_variable(ds, "sol_eff_irr", "Solar effective Irradiance", "W*m-2", standard_name="solar_irradiance_vis")

        u_sol_eff_irr = ds.variables["u_sol_eff_irr"]
        self.assertEqual((), u_sol_eff_irr.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sol_eff_irr.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sol_eff_irr.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar effective Irradiance", u_sol_eff_irr.attrs["long_name"])
        self.assertEqual("Wm^-2", u_sol_eff_irr.attrs["units"])
        self.assertEqual("rectangle", u_sol_eff_irr.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_sol_eff_irr.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_sol_eff_irr.attrs["scan_correlation_scales"])
        self.assertEqual("rectangle", u_sol_eff_irr.attrs["time_correlation_form"])
        self.assertEqual("line", u_sol_eff_irr.attrs["time_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_sol_eff_irr.attrs["time_correlation_scales"])
        self.assertEqual("rectangle", u_sol_eff_irr.attrs["image_correlation_form"])
        self.assertEqual("days", u_sol_eff_irr.attrs["image_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_sol_eff_irr.attrs["image_correlation_scales"])
        self.assertEqual("rectangle", u_sol_eff_irr.attrs["pdf_shape"])

        srf = ds.variables["spectral_response_function_vis"]
        self.assertEqual((1011,), srf.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), srf.data[116])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), srf.attrs["_FillValue"])
        self.assertEqual("Spectral Response Function", srf.attrs["long_name"])

        cov_srf = ds.variables["covariance_spectral_response_function_vis"]
        self.assertEqual((1011, 1011), cov_srf.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), cov_srf.data[116, 22])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), cov_srf.attrs["_FillValue"])
        self.assertEqual("Covariance of the Visible Band Spectral Response Function", cov_srf.attrs["long_name"])

        self._assert_scalar_float_variable(ds, "a_ir", "Calibration parameter a for IR Band", "mWm^-2sr^-1cm^-1")
        self._assert_scalar_float_variable(ds, "b_ir", "Calibration parameter b for IR Band", "mWm^-2sr^-1cm^-1/DC")
        self._assert_scalar_float_variable(ds, "u_a_ir", "Uncertainty of calibration parameter a for IR Band", "mWm^-2sr^-1cm^-1")
        self._assert_scalar_float_variable(ds, "u_b_ir", "Uncertainty of calibration parameter b for IR Band", "mWm^-2sr^-1cm^-1/DC")
        self._assert_scalar_float_variable(ds, "q_wv", "WV Band Calibration quality flag", "1")
        self._assert_scalar_float_variable(ds, "unit_conversion_ir", "IR Unit conversion factor", "1")
        self._assert_scalar_float_variable(ds, "unit_conversion_wv", "WV Unit conversion factor", "1")
        self._assert_scalar_float_variable(ds, "bt_a_ir", "IR Band BT conversion parameter A", "1")
        self._assert_scalar_float_variable(ds, "bt_b_ir", "IR Band BT conversion parameter B", "1")
        self._assert_scalar_float_variable(ds, "bt_a_wv", "WV Band BT conversion parameter A", "1")
        self._assert_scalar_float_variable(ds, "bt_b_wv", "WV Band BT conversion parameter B", "1")

    def test_get_swath_width(self):
        self.assertEqual(5000, MVIRI.get_swath_width())

    def test_add_easy_fcdr_variables(self):
        ds = xr.Dataset()
        MVIRI.add_easy_fcdr_variables(ds, 8)

        reflectance = ds.variables["reflectance"]
        self.assertEqual((5000, 5000), reflectance.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), reflectance.data[3, 115])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), reflectance.attrs["_FillValue"])
        self.assertEqual("toa_bidirectional_reflectance", reflectance.attrs["standard_name"])
        self.assertEqual("percent", reflectance.attrs["units"])
        self.assertEqual(1.52588E-05, reflectance.attrs["scale_factor"])

        u_random = ds.variables["u_random"]
        self.assertEqual((5000, 5000), u_random.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_random.data[118, 234])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_random.attrs["_FillValue"])
        self.assertEqual("random uncertainty per pixel", u_random.attrs["long_name"])
        self.assertEqual(1.52588E-05, u_random.attrs["scale_factor"])
        self.assertEqual("percent", u_random.attrs["units"])

        u_non_random = ds.variables["u_non_random"]
        self.assertEqual((5000, 5000), u_non_random.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_non_random.data[118, 234])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_non_random.attrs["_FillValue"])
        self.assertEqual("non-random uncertainty per pixel", u_non_random.attrs["long_name"])
        self.assertEqual(1.52588E-05, u_non_random.attrs["scale_factor"])
        self.assertEqual("percent", u_non_random.attrs["units"])

    def test_add_full_fcdr_variables(self):
        ds = xr.Dataset()
        MVIRI.add_full_fcdr_variables(ds, 7)

        u_toa_refl_vis = ds.variables["u_toa_bidirectional_reflectance_vis"]
        self.assertEqual((5000, 5000), u_toa_refl_vis.shape)
        self.assertEqual(65535, u_toa_refl_vis.data[6, 110])
        self.assertEqual(65535, u_toa_refl_vis.attrs["_FillValue"])
        self.assertEqual("percent", u_toa_refl_vis.attrs["units"])
        self.assertEqual(1E-03, u_toa_refl_vis.attrs["scale_factor"])
        self.assertEqual("truncated_gaussian", u_toa_refl_vis.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_toa_refl_vis.attrs["scan_correlation_units"])
        self.assertEqual([-2, 2], u_toa_refl_vis.attrs["scan_correlation_scales"])
        self.assertEqual("truncated_gaussian", u_toa_refl_vis.attrs["time_correlation_form"])
        self.assertEqual("line", u_toa_refl_vis.attrs["time_correlation_units"])
        self.assertEqual([-2, 2], u_toa_refl_vis.attrs["time_correlation_scales"])
        self.assertEqual("triangle", u_toa_refl_vis.attrs["image_correlation_form"])
        self.assertEqual("months", u_toa_refl_vis.attrs["image_correlation_units"])
        self.assertEqual([-3, 3], u_toa_refl_vis.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_toa_refl_vis.attrs["pdf_shape"])

        count = ds.variables["count_vis"]
        self.assertEqual((5000, 5000), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.data[0, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint8), count.attrs["_FillValue"])
        self.assertEqual("Image counts", count.attrs["long_name"])
        self.assertEqual("count", count.attrs["units"])

        u_lat = ds.variables["u_latitude"]
        self.assertEqual((5000, 5000), u_lat.shape)
        self.assertEqual(65535, u_lat.data[5, 109])
        self.assertEqual(65535, u_lat.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Latitude", u_lat.attrs["long_name"])
        self.assertEqual("degree", u_lat.attrs["units"])
        self.assertEqual(7.62939E-05, u_lat.attrs["scale_factor"])
        self.assertEqual("triangle", u_lat.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_lat.attrs["scan_correlation_units"])
        self.assertEqual([-250, 250], u_lat.attrs["scan_correlation_scales"])
        self.assertEqual("triangle", u_lat.attrs["time_correlation_form"])
        self.assertEqual("line", u_lat.attrs["time_correlation_units"])
        self.assertEqual([-250, 250], u_lat.attrs["time_correlation_scales"])
        self.assertEqual("triangle", u_lat.attrs["image_correlation_form"])
        self.assertEqual("images", u_lat.attrs["image_correlation_units"])
        self.assertEqual([-12, 0], u_lat.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_lat.attrs["pdf_shape"])

        u_lon = ds.variables["u_longitude"]
        self.assertEqual((5000, 5000), u_lon.shape)
        self.assertEqual(65535, u_lon.data[6, 110])
        self.assertEqual(65535, u_lon.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Longitude", u_lon.attrs["long_name"])
        self.assertEqual("degree", u_lon.attrs["units"])
        self.assertEqual(7.62939E-05, u_lon.attrs["scale_factor"])
        self.assertEqual("triangle", u_lon.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_lon.attrs["scan_correlation_units"])
        self.assertEqual([-250, 250], u_lon.attrs["scan_correlation_scales"])
        self.assertEqual("triangle", u_lon.attrs["time_correlation_form"])
        self.assertEqual("line", u_lon.attrs["time_correlation_units"])
        self.assertEqual([-250, 250], u_lon.attrs["time_correlation_scales"])
        self.assertEqual("triangle", u_lon.attrs["image_correlation_form"])
        self.assertEqual("images", u_lon.attrs["image_correlation_units"])
        self.assertEqual([-12, 0], u_lon.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_lon.attrs["pdf_shape"])

        u_time = ds.variables["u_time"]
        self.assertEqual((5000, 5000), u_time.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_time.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_time.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Time", u_time.attrs["standard_name"])
        self.assertEqual("s", u_time.attrs["units"])
        self.assertEqual(0.009155273, u_time.attrs["scale_factor"])
        self.assertEqual("rectangle", u_time.attrs["pdf_shape"])

        u_sat_zenith = ds.variables["u_satellite_zenith_angle"]
        self.assertEqual((5000, 5000), u_sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sat_zenith.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sat_zenith.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Satellite Zenith Angle", u_sat_zenith.attrs["long_name"])
        self.assertEqual("degree", u_sat_zenith.attrs["units"])
        self.assertEqual(7.62939E-05, u_sat_zenith.attrs["scale_factor"])

        u_sat_azimuth = ds.variables["u_satellite_azimuth_angle"]
        self.assertEqual((5000, 5000), u_sat_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sat_azimuth.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sat_azimuth.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Satellite Azimuth Angle", u_sat_azimuth.attrs["long_name"])
        self.assertEqual("degree", u_sat_azimuth.attrs["units"])
        self.assertEqual(7.62939E-05, u_sat_azimuth.attrs["scale_factor"])

        u_sol_zenith = ds.variables["u_solar_zenith_angle"]
        self.assertEqual((5000, 5000), u_sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sol_zenith.data[1, 112])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sol_zenith.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar Zenith Angle", u_sol_zenith.attrs["long_name"])
        self.assertEqual("degree", u_sol_zenith.attrs["units"])
        self.assertEqual(7.62939E-05, u_sol_zenith.attrs["scale_factor"])

        u_sol_azimuth = ds.variables["u_solar_azimuth_angle"]
        self.assertEqual((5000, 5000), u_sol_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sol_azimuth.data[2, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_sol_azimuth.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar Azimuth Angle", u_sol_azimuth.attrs["long_name"])
        self.assertEqual("degree", u_sol_azimuth.attrs["units"])
        self.assertEqual(7.62939E-05, u_sol_azimuth.attrs["scale_factor"])

        u_tot_count = ds.variables["u_combined_counts_vis"]
        self.assertEqual((5000, 5000), u_tot_count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_tot_count.data[2, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_tot_count.attrs["_FillValue"])
        self.assertEqual("Total Uncertainty in counts", u_tot_count.attrs["long_name"])
        self.assertEqual("count", u_tot_count.attrs["units"])
        self.assertEqual(7.62939E-05, u_tot_count.attrs["scale_factor"])
        self.assertEqual("eiffel", u_tot_count.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_tot_count.attrs["scan_correlation_units"])
        self.assertEqual([-2, 2], u_tot_count.attrs["scan_correlation_scales"])
        self.assertEqual("eiffel", u_tot_count.attrs["time_correlation_form"])
        self.assertEqual("line", u_tot_count.attrs["time_correlation_units"])
        self.assertEqual([-2, 2], u_tot_count.attrs["time_correlation_scales"])
        self.assertEqual("digitised_gaussian", u_tot_count.attrs["pdf_shape"])

        u_srf = ds.variables["u_srf"]
        self.assertEqual((1011, 1011), u_srf.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_srf.data[3, 119])
        self.assertEqual(DefaultData.get_default_fill_value(np.uint16), u_srf.attrs["_FillValue"])
        self.assertEqual("Uncertainty in SRF", u_srf.attrs["long_name"])
        self.assertEqual(1.52588E-05, u_srf.attrs["scale_factor"])
        self.assertEqual("rectangle", u_srf.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_srf.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_srf.attrs["scan_correlation_scales"])
        self.assertEqual("rectangle", u_srf.attrs["time_correlation_form"])
        self.assertEqual("line", u_srf.attrs["time_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_srf.attrs["time_correlation_scales"])
        self.assertEqual("rectangle", u_srf.attrs["image_correlation_form"])
        self.assertEqual("days", u_srf.attrs["image_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_srf.attrs["image_correlation_scales"])
        self.assertEqual("rectangle", u_srf.attrs["pdf_shape"])

        a0 = ds.variables["a0_vis"]
        self.assertEqual((), a0.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a0.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a0.attrs["_FillValue"])
        self.assertEqual("Calibration Coefficient at Launch", a0.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count", a0.attrs["units"])

        a1 = ds.variables["a1_vis"]
        self.assertEqual((), a1.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a1.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a1.attrs["_FillValue"])
        self.assertEqual("Time variation of a0", a1.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count day^-1 10^5", a1.attrs["units"])

        k_space = ds.variables["mean_counts_space_vis"]
        self.assertEqual((), k_space.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), k_space.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), k_space.attrs["_FillValue"])
        self.assertEqual("Space count", k_space.attrs["long_name"])
        self.assertEqual("count", k_space.attrs["units"])

        u_a0_vis = ds.variables["u_a0_vis"]
        self.assertEqual((), u_a0_vis.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a0_vis.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a0_vis.attrs["_FillValue"])
        self.assertEqual("Uncertainty in a0", u_a0_vis.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count", u_a0_vis.attrs["units"])
        self.assertEqual("rectangle", u_a0_vis.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_a0_vis.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_a0_vis.attrs["scan_correlation_scales"])
        self.assertEqual("rectangle", u_a0_vis.attrs["time_correlation_form"])
        self.assertEqual("line", u_a0_vis.attrs["time_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_a0_vis.attrs["time_correlation_scales"])
        self.assertEqual("triangle", u_a0_vis.attrs["image_correlation_form"])
        self.assertEqual("months", u_a0_vis.attrs["image_correlation_units"])
        self.assertEqual([-1.5, 1.5], u_a0_vis.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_a0_vis.attrs["pdf_shape"])

        u_a1_vis = ds.variables["u_a1_vis"]
        self.assertEqual((), u_a0_vis.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a1_vis.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a1_vis.attrs["_FillValue"])
        self.assertEqual("Uncertainty in a1", u_a1_vis.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count day^-1 10^5", u_a1_vis.attrs["units"])
        self.assertEqual("rectangle", u_a1_vis.attrs["scan_correlation_form"])
        self.assertEqual("pixel", u_a1_vis.attrs["scan_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_a1_vis.attrs["scan_correlation_scales"])
        self.assertEqual("rectangle", u_a1_vis.attrs["time_correlation_form"])
        self.assertEqual("line", u_a1_vis.attrs["time_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_a1_vis.attrs["time_correlation_scales"])
        self.assertEqual("triangle", u_a1_vis.attrs["image_correlation_form"])
        self.assertEqual("months", u_a1_vis.attrs["image_correlation_units"])
        self.assertEqual([-1.5, 1.5], u_a1_vis.attrs["image_correlation_scales"])
        self.assertEqual("gaussian", u_a1_vis.attrs["pdf_shape"])

        u_a0_a1_cov = ds.variables["covariance_a0_a1_vis"]
        self.assertEqual((2, 2), u_a0_a1_cov.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a0_a1_cov.data[1, 0])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a0_a1_cov.attrs["_FillValue"])
        self.assertEqual("Covariance matrix of calibration coefficients", u_a0_a1_cov.attrs["long_name"])

        u_e_noise = ds.variables["u_electronics_counts_vis"]
        self.assertEqual((), u_e_noise.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_e_noise.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_e_noise.attrs["_FillValue"])
        self.assertEqual("Uncertainty due to Electronics noise", u_e_noise.attrs["long_name"])
        self.assertEqual("count", u_e_noise.attrs["units"])

        u_digitization = ds.variables["u_digitization_counts_vis"]
        self.assertEqual((), u_digitization.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_digitization.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_digitization.attrs["_FillValue"])
        self.assertEqual("Uncertainty due to digitization", u_digitization.attrs["long_name"])
        self.assertEqual("count", u_digitization.attrs["units"])

        u_space = ds.variables["allan_deviation_counts_space_vis"]
        self.assertEqual((), u_space.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_space.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_space.attrs["_FillValue"])
        self.assertEqual("Uncertainty of space count", u_space.attrs["long_name"])
        self.assertEqual("count", u_space.attrs["units"])
        self.assertEqual("rectangle", u_space.attrs["time_correlation_form"])
        self.assertEqual("line", u_space.attrs["time_correlation_units"])
        self.assertEqual([-np.inf, np.inf], u_space.attrs["time_correlation_scales"])
        self.assertEqual("digitised_gaussian", u_space.attrs["pdf_shape"])

    def _assert_scalar_float_variable(self, ds, name, long_name, units, standard_name=None):
        dse = ds.variables[name]
        self.assertEqual((), dse.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), dse.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), dse.attrs["_FillValue"])

        if standard_name is not None:
            self.assertEqual(standard_name, dse.attrs["standard_name"])

        self.assertEqual(long_name, dse.attrs["long_name"])
        self.assertEqual(units, dse.attrs["units"])
