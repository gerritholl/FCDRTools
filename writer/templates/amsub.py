from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil

BTEMPS_FILL_VALUE = -999999


class AMSUB:
    @staticmethod
    def add_original_variables(dataset, width, height):
        TemplateUtil.add_geolocation_variables(dataset, width, height)

        default_array = DefaultData.create_default_array_3d(width, height, 5, int, BTEMPS_FILL_VALUE)
        variable = Variable(["channel", "y", "x"], default_array)
        variable.attrs["_FillValue"] = BTEMPS_FILL_VALUE
        variable.attrs["standard_name"] = "toa_brightness_temperature"
        variable.attrs["units"] = "K"
        variable.attrs["scale_factor"] = 0.01
        dataset["btemps"] = variable
