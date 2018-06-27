import numpy as np
from xarray import Variable

from fiduceo.common.writer.default_data import DefaultData
from fiduceo.common.writer.templates.templateutil import TemplateUtil as tu


class UTH:

    @staticmethod
    def add_variables(dataset, width, height):
        tu.add_gridded_geolocation_variables(dataset, width, height)
        tu.add_quality_flags(dataset, width, height)

        dataset["time_ranges_ascending"] = UTH._create_time_ranges_variable(height, width, "Minimum and maximum seconds of day pixel contribution time, ascending nodes")
        dataset["time_ranges_descending"] = UTH._create_time_ranges_variable(height, width, "Minimum and maximum seconds of day pixel contribution time, descending nodes")

        dataset["observation_count_ascending"] = UTH._create_observation_counts_variable(height, width, "Number of observations contributing to pixel value, ascending nodes")
        dataset["observation_count_descending"] = UTH._create_observation_counts_variable(height, width, "Number of observations contributing to pixel value, descending nodes")

        dataset["uth_ascending"] = UTH._create_uth_variable(height, width)
        dataset["uth_descending"] = UTH._create_uth_variable(height, width)

        dataset["u_independent_uth_ascending"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to independent effects, ascending nodes", coordinates="lon lat")
        dataset["u_independent_uth_descending"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to independent effects, descending nodes", coordinates="lon lat")
        dataset["u_structured_uth_ascending"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to structured effects, ascending nodes", coordinates="lon lat")
        dataset["u_structured_uth_descending"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to structured effects, descending nodes", coordinates="lon lat")
        dataset["u_common_uth_ascending"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to common effects, ascending nodes", coordinates="lon lat")
        dataset["u_common_uth_descending"] = tu.create_CDR_uncertainty(width, height, "Uncertainty of uth due to common effects, descending nodes", coordinates="lon lat")

    @staticmethod
    def _create_uth_variable(height, width):
        default_array = DefaultData.create_default_array(width, height, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, np.NaN)
        variable.attrs["coordinates"] = "lon lat"
        return variable

    @staticmethod
    def _create_observation_counts_variable(height, width, description):
        fill_value = DefaultData.get_default_fill_value(np.int16)
        default_array = DefaultData.create_default_array(width, height, np.int16, fill_value=fill_value)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, fill_value)
        variable.attrs["description"] = description
        variable.attrs["coordinates"] = "lon lat"
        return variable

    @staticmethod
    def _create_time_ranges_variable(height, width, description):
        default_array = DefaultData.create_default_array_3d(width, height, 2, np.int32, fill_value=4294967295)
        variable = Variable(["bounds", "y", "x"], default_array)
        tu.add_fill_value(variable, 4294967295)
        tu.add_units(variable, "s")
        variable.attrs["description"] = description
        variable.attrs["coordinates"] = "lon lat"
        return variable
