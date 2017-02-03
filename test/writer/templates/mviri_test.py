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

        sat_zenith = ds.variables["satellite_zenith_angle"]
        self.assertEqual((5000, 5000), sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.data[0, 110])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sat_zenith.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_zenith.attrs["standard_name"])
        self.assertEqual("degree", sat_zenith.attrs["units"])

        sol_azimuth = ds.variables["solar_azimuth_angle"]
        self.assertEqual((5000, 5000), sol_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_azimuth.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_azimuth.attrs["_FillValue"])
        self.assertEqual("solar_azimuth_angle", sol_azimuth.attrs["standard_name"])
        self.assertEqual("degree", sol_azimuth.attrs["units"])

        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertEqual((5000, 5000), sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.data[0, 112])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), sol_zenith.attrs["_FillValue"])
        self.assertEqual("solar_zenith_angle", sol_zenith.attrs["standard_name"])
        self.assertEqual("degree", sol_zenith.attrs["units"])

        count = ds.variables["count"]
        self.assertEqual((5000, 5000), count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), count.data[0, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), count.attrs["_FillValue"])
        self.assertEqual("Image counts", count.attrs["standard_name"])
        self.assertEqual("count", count.attrs["units"])

    def test_get_swath_width(self):
        self.assertEqual(5000, MVIRI.get_swath_width())

    def test_add_uncertainty_variables(self):
        ds = xr.Dataset()
        MVIRI.add_uncertainty_variables(ds, 7)

        srf = ds.variables["srf"]
        self.assertEqual((176,), srf.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), srf.data[114])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), srf.attrs["_FillValue"])
        self.assertEqual("Spectral Response Function", srf.attrs["standard_name"])

        a0 = ds.variables["a0"]
        self.assertEqual((5000, 5000), a0.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a0.data[2, 114])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a0.attrs["_FillValue"])
        self.assertEqual("Calibration Coefficient at Launch", a0.attrs["standard_name"])

        a1 = ds.variables["a1"]
        self.assertEqual((5000, 5000), a1.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a1.data[3, 115])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), a1.attrs["_FillValue"])
        self.assertEqual("Time variation of a0", a1.attrs["standard_name"])

        sol_irr = ds.variables["sol_irr"]
        self.assertEqual((24,), sol_irr.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_irr.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), sol_irr.attrs["_FillValue"])
        self.assertEqual("Solar Irradiance", sol_irr.attrs["standard_name"])

        u_lat = ds.variables["u_lat"]
        self.assertEqual((5000, 5000), u_lat.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_lat.data[5, 109])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_lat.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Latitude", u_lat.attrs["standard_name"])
        self.assertEqual("degree", u_lat.attrs["units"])

        u_lon = ds.variables["u_lon"]
        self.assertEqual((5000, 5000), u_lon.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_lon.data[6, 110])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_lon.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Longitude", u_lon.attrs["standard_name"])
        self.assertEqual("degree", u_lon.attrs["units"])

        u_time = ds.variables["u_time"]
        self.assertEqual((5000, 5000), u_time.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_time.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_time.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Time", u_time.attrs["standard_name"])
        self.assertEqual("s", u_time.attrs["units"])

        u_sat_zenith = ds.variables["u_satellite_zenith_angle"]
        self.assertEqual((5000, 5000), u_sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sat_zenith.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sat_zenith.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Satellite Zenith Angle", u_sat_zenith.attrs["standard_name"])
        self.assertEqual("degree", u_sat_zenith.attrs["units"])

        u_sat_azimuth = ds.variables["u_satellite_azimuth_angle"]
        self.assertEqual((5000, 5000), u_sat_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sat_azimuth.data[0, 111])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sat_azimuth.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Satellite Azimuth Angle", u_sat_azimuth.attrs["standard_name"])
        self.assertEqual("degree", u_sat_azimuth.attrs["units"])

        u_sol_zenith = ds.variables["u_solar_zenith_angle"]
        self.assertEqual((5000, 5000), u_sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_zenith.data[1, 112])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_zenith.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar Zenith Angle", u_sol_zenith.attrs["standard_name"])
        self.assertEqual("degree", u_sol_zenith.attrs["units"])

        u_sol_azimuth = ds.variables["u_solar_azimuth_angle"]
        self.assertEqual((5000, 5000), u_sol_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_azimuth.data[2, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), u_sol_azimuth.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar Azimuth Angle", u_sol_azimuth.attrs["standard_name"])
        self.assertEqual("degree", u_sol_azimuth.attrs["units"])

        u_tot_count = ds.variables["u_tot_count"]
        self.assertEqual((5000, 5000), u_tot_count.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_tot_count.data[2, 113])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_tot_count.attrs["_FillValue"])
        self.assertEqual("Total Uncertainty in counts", u_tot_count.attrs["standard_name"])
        self.assertEqual("count", u_tot_count.attrs["units"])

        u_srf = ds.variables["u_srf"]
        self.assertEqual((176,), u_srf.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_srf.data[116])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_srf.attrs["_FillValue"])
        self.assertEqual("Uncertainty in SRF", u_srf.attrs["standard_name"])

        u_a0 = ds.variables["u_a0"]
        self.assertEqual((5000, 5000), u_a0.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a0.data[5, 117])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a0.attrs["_FillValue"])
        self.assertEqual("Uncertainty in a0", u_a0.attrs["standard_name"])

        u_a1 = ds.variables["u_a1"]
        self.assertEqual((5000, 5000), u_a1.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a1.data[6, 118])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_a1.attrs["_FillValue"])
        self.assertEqual("Uncertainty in a1", u_a1.attrs["standard_name"])

        u_sol_irr = ds.variables["u_sol_irr"]
        self.assertEqual((24,), u_sol_irr.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sol_irr.data[7])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sol_irr.attrs["_FillValue"])
        self.assertEqual("Uncertainty in Solar Irradiance", u_sol_irr.attrs["standard_name"])
