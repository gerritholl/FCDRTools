import numpy as np
from xarray import Variable

from fiduceo.fcdr.writer.templates.hirs import HIRS
from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.templateutil import TemplateUtil as tu


class HIRS2(HIRS):
    @staticmethod
    def add_original_variables(dataset, height):
        HIRS.add_geolocation_variables(dataset, height)

        HIRS.add_bt_variable(dataset, height)
        HIRS2._add_angle_variables(dataset, height)

        HIRS.add_original_variables(dataset, height)

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        HIRS.add_easy_fcdr_variables(dataset, height)

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        HIRS.add_full_fcdr_variables(dataset, height)

    @staticmethod
    def get_swath_width():
        return HIRS.get_swath_width()

    @staticmethod
    def _add_angle_variables(dataset, height):
        default_array = DefaultData.create_default_vector(height, np.uint16)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.uint16))
        variable.attrs["standard_name"] = "platform_zenith_angle"
        tu.add_units(variable, "degree")
        tu.add_scale_factor(variable, 0.01)
        tu.add_offset(variable, -180.0)
        dataset["satellite_zenith_angle"] = variable

        dataset["solar_azimuth_angle"] = HIRS._create_geo_angle_variable("solar_azimuth_angle", height)


