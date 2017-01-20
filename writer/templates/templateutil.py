from xarray import Variable

from writer.default_data import DefaultData


class TemplateUtil:
    @staticmethod
    def add_geolocation_variables(dataset, width, height):
        default_array = DefaultData.create_default_array(width, height, float, -32768.0)

        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = -32768.0
        variable.attrs["standard_name"] = "latitude"
        variable.attrs["units"] = "degrees_north"
        dataset["latitude"] = variable

        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = -32768.0
        variable.attrs["standard_name"] = "longitude"
        variable.attrs["units"] = "degrees_east"
        dataset["longitude"] = variable
