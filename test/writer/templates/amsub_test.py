import unittest

import numpy as np
import xarray as xr

from writer.default_data import DefaultData
from writer.templates.amsub import AMSUB


class AMSUBTest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        AMSUB.add_original_variables(ds, 4)

        latitude = ds.variables["latitude"]
        self.assertEqual((4, 90), latitude.shape)
        self.assertEqual(-32768.0, latitude.data[0, 0])
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])

        longitude = ds.variables["longitude"]
        self.assertEqual((4, 90), longitude.shape)
        self.assertEqual(-32768.0, longitude.data[1, 0])
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])

        btemps = ds.variables["btemps"]
        self.assertEqual((5, 4, 90), btemps.shape)
        self.assertEqual(-999999, btemps.data[0, 2, 0])
        self.assertEqual(-999999, btemps.attrs["_FillValue"])
        self.assertEqual("toa_brightness_temperature", btemps.attrs["standard_name"])
        self.assertEqual("K", btemps.attrs["units"])
        self.assertEqual(0.01, btemps.attrs["scale_factor"])
        self.assertEqual("chanqual qualind scanqual", btemps.attrs["ancillary_variables"])

        chanqual = ds.variables["chanqual"]
        self.assertEqual((5, 4), chanqual.shape)
        self.assertEqual(0, chanqual.data[0, 3])
        self.assertEqual("status_flag", chanqual.attrs["standard_name"])

        instrtemp = ds.variables["instrtemp"]
        self.assertEqual((4,), instrtemp.shape)
        self.assertEqual(-2147483647, instrtemp.data[0])
        self.assertEqual(-2147483647, instrtemp.attrs["_FillValue"])
        self.assertEqual("K", instrtemp.attrs["units"])
        self.assertEqual("instrument_temperature", instrtemp.attrs["long_name"])
        self.assertEqual(0.01, instrtemp.attrs["scale_factor"])

        qualind = ds.variables["qualind"]
        self.assertEqual((4,), qualind.shape)
        self.assertEqual(0, qualind.data[1])
        self.assertEqual("status_flag", qualind.attrs["standard_name"])

        scanqual = ds.variables["scanqual"]
        self.assertEqual((4,), scanqual.shape)
        self.assertEqual(0, scanqual.data[1])
        self.assertEqual("status_flag", scanqual.attrs["standard_name"])

        scnlin = ds.variables["scnlin"]
        self.assertEqual((4,), scnlin.shape)
        self.assertEqual(-2147483647, scnlin.data[2])
        self.assertEqual(-2147483647, scnlin.attrs["_FillValue"])
        self.assertEqual("scanline", scnlin.attrs["long_name"])

        scnlindy = ds.variables["scnlindy"]
        self.assertEqual((4,), scnlindy.shape)
        self.assertEqual(-2147483647, scnlindy.data[3])
        self.assertEqual(-2147483647, scnlindy.attrs["_FillValue"])
        self.assertEqual("Acquisition day of year of scan", scnlindy.attrs["long_name"])

        scnlintime = ds.variables["scnlintime"]
        self.assertEqual((4,), scnlintime.shape)
        self.assertEqual(-2147483647, scnlintime.data[0])
        self.assertEqual(-2147483647, scnlintime.attrs["_FillValue"])
        self.assertEqual("Acquisition time of scan in milliseconds since beginning of the day",
                         scnlintime.attrs["long_name"])
        self.assertEqual("ms", scnlintime.attrs["units"])

        scnlinyr = ds.variables["scnlinyr"]
        self.assertEqual((4,), scnlinyr.shape)
        self.assertEqual(-2147483647, scnlinyr.data[1])
        self.assertEqual(-2147483647, scnlinyr.attrs["_FillValue"])
        self.assertEqual("Acquisition year of scan", scnlinyr.attrs["long_name"])

        sat_azimuth = ds.variables["satellite_azimuth_angle"]
        self.assertEqual((4, 90), sat_azimuth.shape)
        self.assertEqual(-999999, sat_azimuth.data[2, 1])
        self.assertEqual(-999999, sat_azimuth.attrs["_FillValue"])
        self.assertEqual("sensor_azimuth_angle", sat_azimuth.attrs["standard_name"])
        self.assertEqual(0.01, sat_azimuth.attrs["scale_factor"])
        self.assertEqual("degree", sat_azimuth.attrs["units"])

        sat_zenith = ds.variables["satellite_zenith_angle"]
        self.assertEqual((4, 90), sat_zenith.shape)
        self.assertEqual(-999999, sat_zenith.data[2, 1])
        self.assertEqual(-999999, sat_zenith.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_zenith.attrs["standard_name"])
        self.assertEqual(0.01, sat_zenith.attrs["scale_factor"])
        self.assertEqual("degree", sat_zenith.attrs["units"])

        sol_azimuth = ds.variables["solar_azimuth_angle"]
        self.assertEqual((4, 90), sol_azimuth.shape)
        self.assertEqual(-999999, sol_azimuth.data[3, 0])
        self.assertEqual(-999999, sol_azimuth.attrs["_FillValue"])
        self.assertEqual("solar_azimuth_angle", sol_azimuth.attrs["standard_name"])
        self.assertEqual(0.01, sol_azimuth.attrs["scale_factor"])
        self.assertEqual("degree", sol_azimuth.attrs["units"])

        sol_zenith = ds.variables["solar_zenith_angle"]
        self.assertEqual((4, 90), sol_zenith.shape)
        self.assertEqual(-999999, sol_zenith.data[3, 0])
        self.assertEqual(-999999, sol_zenith.attrs["_FillValue"])
        self.assertEqual("solar_zenith_angle", sol_zenith.attrs["standard_name"])
        self.assertEqual(0.01, sol_zenith.attrs["scale_factor"])
        self.assertEqual("degree", sol_zenith.attrs["units"])

        acquisition_time = ds.variables["acquisition_time"]
        self.assertEqual((4,), acquisition_time.shape)
        self.assertEqual(-2147483647, acquisition_time.data[2])
        self.assertEqual(-2147483647, acquisition_time.attrs["_FillValue"])
        self.assertEqual("time", acquisition_time.attrs["standard_name"])
        self.assertEqual("Acquisition time in seconds since 1970-01-01 00:00:00", acquisition_time.attrs["long_name"])
        self.assertEqual("s", acquisition_time.attrs["units"])

    def test_get_swath_width(self):
        self.assertEqual(90, AMSUB.get_swath_width())

    def test_add_uncertainty_variables(self):
        ds = xr.Dataset()
        AMSUB.add_uncertainty_variables(ds, 4)

        u_btemps = ds.variables["u_btemps"]
        self.assertEqual((5, 4, 90), u_btemps.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_btemps.data[3, 1, 34])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_btemps.attrs["_FillValue"])
        self.assertEqual("total uncertainty of brightness temperature", u_btemps.attrs["standard_name"])
        self.assertEqual("K", u_btemps.attrs["units"])

        u_syst_btemps = ds.variables["u_syst_btemps"]
        self.assertEqual((5, 4, 90), u_syst_btemps.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_syst_btemps.data[4, 2, 35])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_syst_btemps.attrs["_FillValue"])
        self.assertEqual("systematic uncertainty of brightness temperature", u_syst_btemps.attrs["standard_name"])
        self.assertEqual("K", u_syst_btemps.attrs["units"])

        u_random_btemps = ds.variables["u_random_btemps"]
        self.assertEqual((5, 4, 90), u_random_btemps.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_random_btemps.data[0, 3, 36])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_random_btemps.attrs["_FillValue"])
        self.assertEqual("noise on brightness temperature", u_random_btemps.attrs["standard_name"])
        self.assertEqual("K", u_random_btemps.attrs["units"])

        u_instrtemp = ds.variables["u_instrtemp"]
        self.assertEqual((4,), u_instrtemp.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_instrtemp.data[1])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_instrtemp.attrs["_FillValue"])
        self.assertEqual("uncertainty of instrument temperature", u_instrtemp.attrs["standard_name"])
        self.assertEqual("K", u_instrtemp.attrs["units"])

        u_latitude = ds.variables["u_latitude"]
        self.assertEqual((4, 90), u_latitude.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_latitude.data[2, 38])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_latitude.attrs["_FillValue"])
        self.assertEqual("uncertainty of latitude", u_latitude.attrs["standard_name"])
        self.assertEqual("degree", u_latitude.attrs["units"])

        u_longitude = ds.variables["u_longitude"]
        self.assertEqual((4, 90), u_longitude.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_longitude.data[3, 39])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_longitude.attrs["_FillValue"])
        self.assertEqual("uncertainty of longitude", u_longitude.attrs["standard_name"])
        self.assertEqual("degree", u_longitude.attrs["units"])

        u_sat_azimuth = ds.variables["u_satellite_azimuth_angle"]
        self.assertEqual((4, 90), u_sat_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sat_azimuth.data[0, 40])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sat_azimuth.attrs["_FillValue"])
        self.assertEqual("uncertainty of satellite azimuth angle", u_sat_azimuth.attrs["standard_name"])
        self.assertEqual("degree", u_sat_azimuth.attrs["units"])

        u_sat_zenith = ds.variables["u_satellite_zenith_angle"]
        self.assertEqual((4, 90), u_sat_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sat_zenith.data[1, 41])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sat_zenith.attrs["_FillValue"])
        self.assertEqual("uncertainty of satellite zenith angle", u_sat_zenith.attrs["standard_name"])
        self.assertEqual("degree", u_sat_zenith.attrs["units"])

        u_sol_azimuth = ds.variables["u_solar_azimuth_angle"]
        self.assertEqual((4, 90), u_sol_azimuth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sol_azimuth.data[2, 42])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sol_azimuth.attrs["_FillValue"])
        self.assertEqual("uncertainty of solar azimuth angle", u_sol_azimuth.attrs["standard_name"])
        self.assertEqual("degree", u_sol_azimuth.attrs["units"])

        u_sol_zenith = ds.variables["u_solar_zenith_angle"]
        self.assertEqual((4, 90), u_sol_zenith.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sol_zenith.data[3, 43])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_sol_zenith.attrs["_FillValue"])
        self.assertEqual("uncertainty of solar zenith angle", u_sol_zenith.attrs["standard_name"])
        self.assertEqual("degree", u_sol_zenith.attrs["units"])

