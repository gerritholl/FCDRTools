import unittest

import numpy as np
import xarray as xr

from writer.default_data import DefaultData
from writer.templates.hirs import HIRS


class HIRSTest(unittest.TestCase):
    def test_add_original_variables(self):
        ds = xr.Dataset()
        HIRS.add_original_variables(ds, 6)

        latitude = ds.variables["latitude"]
        self.assertEqual((6, 56), latitude.shape)
        self.assertEqual(-32768.0, latitude.attrs["_FillValue"])
        self.assertEqual(-32768.0, latitude.data[0, 0])
        self.assertEqual("latitude", latitude.attrs["standard_name"])
        self.assertEqual("degrees_north", latitude.attrs["units"])

        longitude = ds.variables["longitude"]
        self.assertEqual((6, 56), latitude.shape)
        self.assertEqual(-32768.0, longitude.attrs["_FillValue"])
        self.assertEqual(-32768.0, longitude.data[0, 0])
        self.assertEqual("longitude", longitude.attrs["standard_name"])
        self.assertEqual("degrees_east", longitude.attrs["units"])

        bt = ds.variables["bt"]
        self.assertEqual((19, 6, 56), bt.shape)
        self.assertEqual(-999.0, bt.data[0, 2, 1])
        self.assertEqual(-999.0, bt.attrs["_FillValue"])
        self.assertEqual("toa_brightness_temperature", bt.attrs["standard_name"])
        self.assertEqual("K", bt.attrs["units"])

        c_earth = ds.variables["c_earth"]
        self.assertEqual((20, 6, 56), c_earth.shape)
        self.assertEqual(99999, c_earth.data[0, 2, 3])
        self.assertEqual(99999, c_earth.attrs["_FillValue"])
        self.assertEqual("counts_Earth", c_earth.attrs["standard_name"])
        self.assertEqual("count", c_earth.attrs["units"])

        l_earth = ds.variables["L_earth"]
        self.assertEqual((20, 6, 56), l_earth.shape)
        self.assertEqual(-999.0, l_earth.data[0, 2, 4])
        self.assertEqual(-999.0, l_earth.attrs["_FillValue"])
        self.assertEqual("toa_outgoing_inband_radiance", l_earth.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", l_earth.attrs["units"])

        sat_za = ds.variables["sat_za"]
        self.assertEqual((6, 56), sat_za.shape)
        self.assertEqual(-999.0, sat_za.data[2, 2])
        self.assertEqual(-999.0, sat_za.attrs["_FillValue"])
        self.assertEqual("sensor_zenith_angle", sat_za.attrs["standard_name"])
        self.assertEqual("degree", sat_za.attrs["units"])

        scanline = ds.variables["scanline"]
        self.assertEqual((6,), scanline.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.data[3])
        self.assertEqual(DefaultData.get_default_fill_value(np.int16), scanline.attrs["_FillValue"])
        self.assertEqual("scanline_number", scanline.attrs["standard_name"])
        self.assertEqual("number", scanline.attrs["units"])

        scnlinf = ds.variables["scnlinf"]
        self.assertEqual((6,), scnlinf.shape)
        self.assertEqual(9, scnlinf.data[4])
        self.assertEqual(9, scnlinf.attrs["_FillValue"])
        self.assertEqual("0, 1, 2, 3", scnlinf.attrs["flag_values"])
        self.assertEqual("earth_view space_view icct_view iwct_view", scnlinf.attrs["flag_meanings"])
        self.assertEqual("scanline_bitfield", scnlinf.attrs["standard_name"])

    def test_get_swath_width(self):
        self.assertEqual(56, HIRS.get_swath_width())

    def test_add_uncertainty_variables(self):
        ds = xr.Dataset()
        HIRS.add_uncertainty_variables(ds, 7)

        u_lat = ds.variables["u_lat"]
        self.assertEqual((7, 56), u_lat.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_lat.data[3, 3])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_lat.attrs["_FillValue"])
        self.assertEqual("uncertainty_latitude", u_lat.attrs["standard_name"])
        self.assertEqual("degree", u_lat.attrs["units"])

        u_lon = ds.variables["u_lon"]
        self.assertEqual((7, 56), u_lon.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_lon.data[4, 4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_lon.attrs["_FillValue"])
        self.assertEqual("uncertainty_longitude", u_lon.attrs["standard_name"])
        self.assertEqual("degree", u_lon.attrs["units"])

        u_time = ds.variables["u_time"]
        self.assertEqual((7, 56), u_time.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_time.data[5, 5])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_time.attrs["_FillValue"])
        self.assertEqual("uncertainty_time", u_time.attrs["standard_name"])
        self.assertEqual("s", u_time.attrs["units"])

        u_c_earth = ds.variables["u_c_earth"]
        self.assertEqual((20, 7, 56), u_c_earth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_c_earth.data[6, 6, 6])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_c_earth.attrs["_FillValue"])
        self.assertEqual("uncertainty_counts_Earth", u_c_earth.attrs["standard_name"])
        self.assertEqual("count", u_c_earth.attrs["units"])

        u_L_earth_random = ds.variables["u_L_earth_random"]
        self.assertEqual((20, 7, 56), u_L_earth_random.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_random.data[7, 0, 7])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_random.attrs["_FillValue"])
        self.assertEqual("uncertainty_radiance_Earth_random", u_L_earth_random.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", u_L_earth_random.attrs["units"])

        u_L_earth_sr = ds.variables["u_L_earth_structuredrandom"]
        self.assertEqual((20, 7, 56), u_L_earth_sr.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_sr.data[8, 1, 8])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_sr.attrs["_FillValue"])
        self.assertEqual("uncertainty_radiance_Earth_structured_random", u_L_earth_sr.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", u_L_earth_sr.attrs["units"])

        u_L_earth_sys = ds.variables["u_L_earth_systematic"]
        self.assertEqual((20, 7, 56), u_L_earth_sys.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_sys.data[9, 2, 9])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_sys.attrs["_FillValue"])
        self.assertEqual("uncertainty_radiance_Earth_systematic", u_L_earth_sys.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", u_L_earth_sys.attrs["units"])

        u_L_earth_total = ds.variables["u_L_earth_total"]
        self.assertEqual((20, 7, 56), u_L_earth_total.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_total.data[10, 3, 10])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_L_earth_total.attrs["_FillValue"])
        self.assertEqual("uncertainty_radiance_Earth_total", u_L_earth_total.attrs["standard_name"])
        self.assertEqual("mW m^-2 sr^-1 cm", u_L_earth_total.attrs["units"])

        S_u_L_earth = ds.variables["S_u_L_earth"]
        self.assertEqual((20, 20), S_u_L_earth.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), S_u_L_earth.data[11, 4])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), S_u_L_earth.attrs["_FillValue"])
        self.assertEqual("covariance_radiance_Earth", S_u_L_earth.attrs["standard_name"])

        u_bt_random = ds.variables["u_bt_random"]
        self.assertEqual((19, 7, 56), u_bt_random.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_random.data[13, 6, 13])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_random.attrs["_FillValue"])
        self.assertEqual("uncertainty_bt_random", u_bt_random.attrs["standard_name"])
        self.assertEqual("K", u_bt_random.attrs["units"])

        u_bt_structuredrandom = ds.variables["u_bt_structuredrandom"]
        self.assertEqual((19, 7, 56), u_bt_structuredrandom.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_structuredrandom.data[14, 0, 14])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_structuredrandom.attrs["_FillValue"])
        self.assertEqual("uncertainty_bt_structured_random", u_bt_structuredrandom.attrs["standard_name"])
        self.assertEqual("K", u_bt_structuredrandom.attrs["units"])

        u_bt_sys = ds.variables["u_bt_systematic"]
        self.assertEqual((19, 7, 56), u_bt_sys.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_sys.data[15, 1, 15])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_sys.attrs["_FillValue"])
        self.assertEqual("uncertainty_bt_systematic", u_bt_sys.attrs["standard_name"])
        self.assertEqual("K", u_bt_sys.attrs["units"])

        u_bt_total = ds.variables["u_bt_total"]
        self.assertEqual((19, 7, 56), u_bt_total.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_total.data[15, 1, 15])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), u_bt_total.attrs["_FillValue"])
        self.assertEqual("uncertainty_bt_total", u_bt_total.attrs["standard_name"])
        self.assertEqual("K", u_bt_total.attrs["units"])

        S_bt = ds.variables["S_bt"]
        self.assertEqual((19,19), S_bt.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), S_bt.data[12, 3])
        self.assertEqual(DefaultData.get_default_fill_value(np.float32), S_bt.attrs["_FillValue"])
        self.assertEqual("covariance_brightness_temperature", S_bt.attrs["standard_name"])

        # @ todo 1 tb/tb missing: calibration coefficients and the associated uncertainties 2017-0-27

        Tc_baseplate = ds.variables["Tc_baseplate"]
        self.assertEqual((7,), Tc_baseplate.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), Tc_baseplate.data[4])
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), Tc_baseplate.attrs["_FillValue"])
        self.assertEqual("temperature_baseplate_counts", Tc_baseplate.attrs["standard_name"])
        self.assertEqual("count", Tc_baseplate.attrs["units"])

        Tc_ch = ds.variables["Tc_ch"]
        self.assertEqual((7,), Tc_ch.shape)
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), Tc_ch.data[5])
        self.assertEqual(DefaultData.get_default_fill_value(np.int32), Tc_ch.attrs["_FillValue"])
        self.assertEqual("temperature_coolerhousing_counts", Tc_ch.attrs["standard_name"])
        self.assertEqual("count", Tc_ch.attrs["units"])

