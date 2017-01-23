import numpy as np
from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil

FILL_VALUE = -999.0
COUNTS_FILL_VALUE = 99999
NUM_CHANNELS = 19
NUM_RAD_CHANNELS = 20
SWATH_WIDTH = 56


class HIRS:
    @staticmethod
    def add_original_variables(dataset, height):
        TemplateUtil.add_geolocation_variables(dataset, SWATH_WIDTH, height)

        # bt
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_CHANNELS, np.float32, FILL_VALUE)
        variable = Variable(["channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = FILL_VALUE
        variable.attrs["standard_name"] = "toa_brightness_temperature"
        variable.attrs["units"] = "K"
        dataset["bt"] = variable

        # c_earth
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_RAD_CHANNELS, np.int32,
                                                            COUNTS_FILL_VALUE, ["rad_channel", "y", "x"])
        variable = Variable(["rad_channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = COUNTS_FILL_VALUE
        variable.attrs["standard_name"] = "counts_Earth"
        variable.attrs["units"] = "count"
        dataset["c_earth"] = variable

        # L_earth
        default_array = DefaultData.create_default_array_3d(SWATH_WIDTH, height, NUM_RAD_CHANNELS, np.float32,
                                                            FILL_VALUE, ["rad_channel", "y", "x"])
        variable = Variable(["rad_channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = FILL_VALUE
        variable.attrs["standard_name"] = "toa_outgoing_inband_radiance"
        variable.attrs["units"] = "mW m^-2 sr^-1 cm"
        dataset["L_earth"] = variable

        # sat_za
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.float32, fill_value=FILL_VALUE)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = FILL_VALUE
        variable.attrs["standard_name"] = "sensor_zenith_angle"
        variable.attrs["units"] = "degree"
        dataset["sat_za"] = variable

        # scanline
        default_array = DefaultData.create_default_vector(height, np.int16)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.int16)
        variable.attrs["standard_name"] = "scanline_number"
        variable.attrs["units"] = "number"
        dataset["scanline"] = variable

        # scnlinf
        default_array = DefaultData.create_default_vector(height, np.int8, fill_value=9)
        variable = Variable(["y"], default_array)
        variable.attrs["_FillValue"] = 9
        variable.attrs["standard_name"] = "scanline_bitfield"
        variable.attrs["flag_values"] = "0, 1, 2, 3"
        variable.attrs["flag_meanings"] = "earth_view space_view icct_view iwct_view"
        dataset["scnlinf"] = variable