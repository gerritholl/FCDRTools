import unittest

import numpy as np
import xarray as xr

from writer.default_data import DefaultData
from writer.templates.mviri import MVIRI


class MVIRITest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        MVIRI.add_original_variables(ds, 5)

        time = ds.variables["time"]
        self.assertEqual((2500,), time.shape)
        self.assertEqual(-2147483647, time.data[4])
        self.assertEqual(-2147483647, time.attrs["_FillValue"])
        self.assertEqual("time", time.attrs["standard_name"])
        self.assertEqual("Acquisition time in seconds since 1970-01-01 00:00:00", time.attrs["long_name"])
        self.assertEqual("true", time.attrs["_Unsigned"])
        self.assertEqual("s", time.attrs["units"])

        timedelta = ds.variables["timedelta"]
        self.assertEqual((2500, 2500), timedelta.shape)
        self.assertEqual(-32767, timedelta.data[2, 108])
        self.assertEqual(-32767, timedelta.attrs["_FillValue"])
        self.assertEqual("time", timedelta.attrs["standard_name"])
        self.assertEqual("Delta time at pixel acquisition against central pixel", timedelta.attrs["long_name"])
        self.assertEqual("s", timedelta.attrs["units"])
        self.assertEqual(0.001831083, timedelta.attrs["scale_factor"])

        sat_azimuth = ds.variables["satellite_azimuth_angle"]
        self.assertEqual((5000, 5000), sat_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_azimuth.data[0, 109])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_azimuth.attrs["_FillValue"])
        self.assertEqual("sensor_azimuth_angle", sat_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sat_azimuth.attrs["units"])
        self.assertEqual(0.005493164, sat_azimuth.attrs["scale_factor"])
        self.assertEqual("true", sat_azimuth.attrs["_Unsigned"])

        sat_zenith = ds.variables["satellite_zenith_angle"]
        self.assertEqual((5000, 5000), sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.data[0, 110])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_zenith.attrs["standard_name"])
        self.assertEqual("degree", sat_zenith.attrs["units"])
        self.assertEqual(0.005493248, sat_zenith.attrs["scale_factor"])

        sol_azimuth = ds.variables["solar_azimuth_angle"]
        self.assertEqual((5000, 5000), sol_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_azimuth.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_azimuth.attrs["_FillValue"])
        self.assertEqual("solar_azimuth_angle", sol_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sol_azimuth.attrs["units"])
        self.assertEqual(0.005493164, sol_azimuth.attrs["scale_factor"])
        self.assertEqual("true", sol_azimuth.attrs["_Unsigned"])

        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertEqual((5000, 5000), sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.data[0, 112])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.attrs["_FillValue"])
        self.assertEqual("solar_zenith_angle", sol_zenith.attrs["standard_name"])
        self.assertEqual("degree", sol_zenith.attrs["units"])
        self.assertEqual(0.005493248, sol_zenith.attrs["scale_factor"])

        count = ds.variables["count"]
        self.assertEqual((5000, 5000), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int8), count.data[0, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.int8), count.attrs["_FillValue"])
        self.assertEqual("Image counts", count.attrs["long_name"])
        self.assertEqual("count", count.attrs["units"])
        self.assertEqual("true", count.attrs["_Unsigned"])

        reflectance = ds.variables["reflectance"]
        self.assertEqual((5000, 5000), reflectance.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), reflectance.data[3, 115])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), reflectance.attrs["_FillValue"])
        self.assertEqual("toa_reflectance", reflectance.attrs["standard_name"])
        self.assertEqual("percent", reflectance.attrs["units"])
        self.assertEqual(1.52588E-05, reflectance.attrs["scale_factor"])
        self.assertEqual("true", count.attrs["_Unsigned"])

        srf = ds.variables["srf"]
        self.assertEqual((1000,), srf.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), srf.data[116])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), srf.attrs["_FillValue"])
        self.assertEqual("Spectral Response Function", srf.attrs["long_name"])

    def test_get_swath_width(self):
        self.assertEqual(5000, MVIRI.get_swath_width())

    def test_add_uncertainty_variables(self):
        ds = xr.Dataset()
        MVIRI.add_uncertainty_variables(ds, 7)

        sol_irr = ds.variables["sol_irr"]
        self.assertEqual((1000,), sol_irr.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_irr.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_irr.attrs["_FillValue"])
        self.assertEqual("solar_irradiance_per_unit_wavelength", sol_irr.attrs["standard_name"])
        self.assertEqual("W*m-2*nm-1", sol_irr.attrs["units"])

        sol_eff_irr = ds.variables["sol_eff_irr"]
        self.assertEqual((), sol_eff_irr.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_eff_irr.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_eff_irr.attrs["_FillValue"])
        self.assertEqual("Solar effective Irradiance", sol_eff_irr.attrs["long_name"])
        self.assertEqual("W*m-2", sol_eff_irr.attrs["units"])

        u_lat = ds.variables["u_latitude"]
        self.assertEqual((5000, 5000), u_lat.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_lat.data[5, 109])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_lat.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Latitude", u_lat.attrs["long_name"])
        self.assertEqual("degree", u_lat.attrs["units"])
        self.assertEqual(7.62939E-05, u_lat.attrs["scale_factor"])
        self.assertEqual("true", u_lat.attrs["_Unsigned"])

        u_lon = ds.variables["u_longitude"]
        self.assertEqual((5000, 5000), u_lon.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_lon.data[6, 110])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_lon.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Longitude", u_lon.attrs["long_name"])
        self.assertEqual("degree", u_lon.attrs["units"])
        self.assertEqual(7.62939E-05, u_lon.attrs["scale_factor"])
        self.assertEqual("true", u_lon.attrs["_Unsigned"])

        u_time = ds.variables["u_time"]
        self.assertEqual((5000, 5000), u_time.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_time.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_time.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Time", u_time.attrs["standard_name"])
        self.assertEqual("s", u_time.attrs["units"])
        self.assertEqual("true", u_time.attrs["_Unsigned"])

        u_sat_zenith = ds.variables["u_satellite_zenith_angle"]
        self.assertEqual((5000, 5000), u_sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sat_zenith.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sat_zenith.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Satellite Zenith Angle", u_sat_zenith.attrs["long_name"])
        self.assertEqual("degree", u_sat_zenith.attrs["units"])
        self.assertEqual(7.62939E-05, u_sat_zenith.attrs["scale_factor"])
        self.assertEqual("true", u_sat_zenith.attrs["_Unsigned"])

        u_sat_azimuth = ds.variables["u_satellite_azimuth_angle"]
        self.assertEqual((5000, 5000), u_sat_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sat_azimuth.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sat_azimuth.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Satellite Azimuth Angle", u_sat_azimuth.attrs["long_name"])
        self.assertEqual("degree", u_sat_azimuth.attrs["units"])
        self.assertEqual(7.62939E-05, u_sat_azimuth.attrs["scale_factor"])
        self.assertEqual("true", u_sat_azimuth.attrs["_Unsigned"])

        u_sol_zenith = ds.variables["u_solar_zenith_angle"]
        self.assertEqual((5000, 5000), u_sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_zenith.data[1, 112])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_zenith.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar Zenith Angle", u_sol_zenith.attrs["long_name"])
        self.assertEqual("degree", u_sol_zenith.attrs["units"])
        self.assertEqual(7.62939E-05, u_sol_zenith.attrs["scale_factor"])
        self.assertEqual("true", u_sol_zenith.attrs["_Unsigned"])

        u_sol_azimuth = ds.variables["u_solar_azimuth_angle"]
        self.assertEqual((5000, 5000), u_sol_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_azimuth.data[2, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_azimuth.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar Azimuth Angle", u_sol_azimuth.attrs["long_name"])
        self.assertEqual("degree", u_sol_azimuth.attrs["units"])
        self.assertEqual(7.62939E-05, u_sol_azimuth.attrs["scale_factor"])
        self.assertEqual("true", u_sol_azimuth.attrs["_Unsigned"])

        u_tot_count = ds.variables["u_tot_count"]
        self.assertEqual((5000, 5000), u_tot_count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_tot_count.data[2, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_tot_count.attrs["_FillValue"])
        self.assertEqual("Total Uncertainty in counts", u_tot_count.attrs["long_name"])
        self.assertEqual("count", u_tot_count.attrs["units"])
        self.assertEqual(7.62939E-05, u_tot_count.attrs["scale_factor"])
        self.assertEqual("true", u_tot_count.attrs["_Unsigned"])

        u_srf = ds.variables["u_srf"]
        self.assertEqual((1000, 1000), u_srf.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_srf.data[3, 119])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_srf.attrs["_FillValue"])
        self.assertEqual("Uncertainty in SRF", u_srf.attrs["long_name"])
        self.assertEqual(1.52588E-05, u_srf.attrs["scale_factor"])
        self.assertEqual("true", u_srf.attrs["_Unsigned"])

        a0 = ds.variables["a0"]
        self.assertEqual((), a0.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a0.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a0.attrs["_FillValue"])
        self.assertEqual("Calibration Coefficient at Launch", a0.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count", a0.attrs["units"])

        a1 = ds.variables["a1"]
        self.assertEqual((), a1.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a1.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a1.attrs["_FillValue"])
        self.assertEqual("Time variation of a0", a1.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count day^-1 10^5", a1.attrs["units"])

        dse = ds.variables["dSE"]
        self.assertEqual((), dse.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int8), dse.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int8), dse.attrs["_FillValue"])
        self.assertEqual("Sun-Earth distance", dse.attrs["long_name"])
        self.assertEqual("au", dse.attrs["units"])
        self.assertEqual(0.00390625, dse.attrs["scale_factor"])
        self.assertEqual("true", dse.attrs["_Unsigned"])

        k_space = ds.variables["K_space"]
        self.assertEqual((), k_space.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int8), k_space.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int8), k_space.attrs["_FillValue"])
        self.assertEqual("Space count", k_space.attrs["long_name"])
        self.assertEqual("count", k_space.attrs["units"])
        self.assertEqual(0.00390625, k_space.attrs["scale_factor"])
        self.assertEqual("true", k_space.attrs["_Unsigned"])

        u_a0 = ds.variables["u_a0"]
        self.assertEqual((), u_a0.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_a0.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_a0.attrs["_FillValue"])
        self.assertEqual("Uncertainty in a0", u_a0.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count", u_a0.attrs["units"])
        self.assertEqual(1.52588E-05, u_a0.attrs["scale_factor"])
        self.assertEqual("true", u_a0.attrs["_Unsigned"])

        u_a1 = ds.variables["u_a1"]
        self.assertEqual((), u_a1.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_a1.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_a1.attrs["_FillValue"])
        self.assertEqual("Uncertainty in a1", u_a1.attrs["long_name"])
        self.assertEqual("Wm^-2sr^-1/count day^-1 10^5", u_a1.attrs["units"])
        self.assertEqual(0.000762939, u_a1.attrs["scale_factor"])
        self.assertEqual("true", u_a1.attrs["_Unsigned"])

        u_sol_eff_irr = ds.variables["u_sol_eff_irr"]
        self.assertEqual((), u_sol_eff_irr.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_eff_irr.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_eff_irr.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar effective Irradiance", u_sol_eff_irr.attrs["long_name"])
        self.assertEqual(0.001525879, u_sol_eff_irr.attrs["scale_factor"])
        self.assertEqual("true", u_sol_eff_irr.attrs["_Unsigned"])

        u_e_noise = ds.variables["u_e-noise"]
        self.assertEqual((), u_e_noise.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_e_noise.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_e_noise.attrs["_FillValue"])
        self.assertEqual("Uncertainty due to Electronics noise", u_e_noise.attrs["long_name"])
        self.assertEqual(7.62939E-05, u_e_noise.attrs["scale_factor"])
        self.assertEqual("count", u_e_noise.attrs["units"])
        self.assertEqual("true", u_e_noise.attrs["_Unsigned"])

        u_shot = ds.variables["u_shot-noise"]
        self.assertEqual((5000, 5000), u_shot.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_shot.data[7, 119])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_shot.attrs["_FillValue"])
        self.assertEqual("Uncertainty due to shot noise", u_shot.attrs["long_name"])
        self.assertEqual("count", u_shot.attrs["units"])
        self.assertEqual("true", u_shot.attrs["_Unsigned"])
        self.assertEqual(7.62939E-05, u_shot.attrs["scale_factor"])

        u_dSE = ds.variables["u_dSE"]
        self.assertEqual((), u_dSE.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_dSE.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_dSE.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Sun-Earth distance", u_dSE.attrs["long_name"])
        self.assertEqual("au", u_dSE.attrs["units"])
        self.assertEqual("true", u_dSE.attrs["_Unsigned"])
        self.assertEqual(7.62939E-05, u_dSE.attrs["scale_factor"])

        u_digitization = ds.variables["u_digitization"]
        self.assertEqual((), u_digitization.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_digitization.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_digitization.attrs["_FillValue"])
        self.assertEqual("Uncertainty due to digitization", u_digitization.attrs["long_name"])
        self.assertEqual(7.62939E-05, u_digitization.attrs["scale_factor"])
        self.assertEqual("count", u_digitization.attrs["units"])
        self.assertEqual("true", u_digitization.attrs["_Unsigned"])

        u_space = ds.variables["u_space"]
        self.assertEqual((), u_space.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_space.data)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_space.attrs["_FillValue"])
        self.assertEqual("Uncertainty of space count", u_space.attrs["long_name"])
        self.assertEqual(7.62939E-05, u_space.attrs["scale_factor"])
        self.assertEqual("count", u_space.attrs["units"])
        self.assertEqual("true", u_space.attrs["_Unsigned"])