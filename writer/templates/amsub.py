from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil


class AMSUB:
    @staticmethod
    def add_original_variables(dataset, height, width):
        TemplateUtil.add_geolocation_variables(dataset, width, height)

        default_array = DefaultData.create_default_array_3d(width, height, 5, int, -999999)
        variable = Variable({"x", "y" , "channel"}, default_array)
       # dataset["btemps"] = variable
