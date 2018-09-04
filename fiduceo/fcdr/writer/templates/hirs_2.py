import numpy as np
from xarray import Variable

from fiduceo.common.writer.templates.templateutil import TemplateUtil as tu
from fiduceo.common.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.hirs import HIRS, CHUNKING_2D

MAX_SRF_SIZE = 102


class HIRS2(HIRS):
    @staticmethod
    def add_original_variables(dataset, height, srf_size=None):
        HIRS.add_geolocation_variables(dataset, height)
        HIRS.add_quality_flags(dataset, height)

        HIRS.add_bt_variable(dataset, height)
        HIRS.add_common_angles(dataset, height)

        if srf_size is None:
            srf_size = MAX_SRF_SIZE

        HIRS.add_common_sensor_variables(dataset, height, srf_size)
        HIRS.add_coordinates(dataset)

    @staticmethod
    def add_easy_fcdr_variables(dataset, height, corr_dx=None, corr_dy=None, lut_size=None):
        HIRS.add_easy_fcdr_variables(dataset, height, corr_dx, corr_dy, lut_size)

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        HIRS.add_full_fcdr_variables(dataset, height)

    @staticmethod
    def get_swath_width():
        return HIRS.get_swath_width()

    @staticmethod
    def add_template_key(dataset):
        dataset.attrs["template_key"] = "HIRS2"
