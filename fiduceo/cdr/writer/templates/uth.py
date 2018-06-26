import numpy as np
from xarray import Variable

from fiduceo.common.writer.default_data import DefaultData
from fiduceo.common.writer.templates.templateutil import TemplateUtil as tu


class UTH:

    @staticmethod
    def add_variables(dataset, width, height):
        tu.add_gridded_geolocation_variables(dataset, width, height)
        tu.add_quality_flags(dataset, width, height)

        default_array = DefaultData.create_default_array_3d(width, height, 2, np.int32, fill_value=4294967295)
        variable = Variable(["bounds", "y", "x"], default_array)
        tu.add_fill_value(variable, 4294967295)
        tu.add_units(variable, "s")
        variable.attrs["description"] = "Minimum and maximum seconds of day pixel contribution time"
        variable.attrs["coordinates"] = "lon lat"
        dataset["time_ranges"] = variable

        fill_value = DefaultData.get_default_fill_value(np.int16)
        default_array = DefaultData.create_default_array(width, height, np.int16, fill_value=fill_value)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, fill_value)
        variable.attrs["description"] = "Number of observations contributing to pixel value"
        variable.attrs["coordinates"] = "lon lat"
        dataset["observation_count"] = variable

        default_array = DefaultData.create_default_array(width, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["coordinates"] = "lon lat"
        dataset["uth"] = variable

        dataset["u_independent_uth"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to independent effects", coordinates="lon lat")
        dataset["u_structured_uth"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to structured effects", coordinates="lon lat")
        dataset["u_common_uth"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to common effects", coordinates="lon lat")
