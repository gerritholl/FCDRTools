import numpy as np
from xarray import Variable

from fiduceo.fcdr.writer.templates.hirs import HIRS, CHUNKING_2D
from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.templateutil import TemplateUtil as tu


class HIRS2(HIRS):
    @staticmethod
    def add_original_variables(dataset, height):
        HIRS.add_geolocation_variables(dataset, height)
        HIRS.add_quality_flags(dataset, height)

        HIRS.add_bt_variable(dataset, height)
        HIRS2._add_angle_variables(dataset, height)

        HIRS.add_common_sensor_variables(dataset, height)
        HIRS.add_coordinates(dataset)

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
    def add_template_key(dataset):
        dataset.attrs["template_key"] = "HIRS2"

    @staticmethod
    def _add_angle_variables(dataset, height):
        default_array = DefaultData.create_default_vector(height, np.float32, fill_value=np.NaN)
        variable = Variable(["y"], default_array)
        variable.attrs["standard_name"] = "platform_zenith_angle"
        tu.add_units(variable, "degree")
        tu.add_encoding(variable, np.uint16, DefaultData.get_default_fill_value(np.uint16), 0.01, -180.0)
        dataset["satellite_zenith_angle"] = variable

        dataset["solar_azimuth_angle"] = HIRS._create_geo_angle_variable("solar_azimuth_angle", height, chunking=CHUNKING_2D)


