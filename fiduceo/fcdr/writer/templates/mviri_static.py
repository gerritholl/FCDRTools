import numpy as np
from xarray import Variable

from fiduceo.fcdr.writer.default_data import DefaultData
from fiduceo.fcdr.writer.templates.mviri import MVIRI
from fiduceo.fcdr.writer.templates.templateutil import TemplateUtil


class MVIRI_STATIC:
    @staticmethod
    def add_original_variables(dataset, height):
        # height is ignored - supplied just for interface compatibility tb 2017-07-19
        # latitude_vis
        default_array = DefaultData.create_default_array(MVIRI.FULL_DIMENSION, MVIRI.FULL_DIMENSION, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "latitude"
        TemplateUtil.add_units(variable, "degrees_north")
        TemplateUtil.add_encoding(variable, np.int16, -32768, scale_factor=0.0027466658)
        dataset["latitude_vis"] = variable

        # longitude_vis
        default_array = DefaultData.create_default_array(MVIRI.FULL_DIMENSION, MVIRI.FULL_DIMENSION, np.float32, fill_value=np.NaN)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["standard_name"] = "longitude"
        TemplateUtil.add_units(variable, "degrees_east")
        TemplateUtil.add_encoding(variable, np.int16, -32768, scale_factor=0.0054933317)
        dataset["longitude_vis"] = variable

        # latitude_ir_wv
        default_array = DefaultData.create_default_array(MVIRI.IR_DIMENSION, MVIRI.IR_DIMENSION, np.float32, fill_value=np.NaN)
        variable = Variable([MVIRI.IR_X_DIMENSION, MVIRI.IR_X_DIMENSION], default_array)
        variable.attrs["standard_name"] = "latitude"
        TemplateUtil.add_units(variable, "degrees_north")
        TemplateUtil.add_encoding(variable, np.int16, -32768, scale_factor=0.0027466658)
        dataset["latitude_ir_wv"] = variable

        # longitude_ir_wv
        default_array = DefaultData.create_default_array(MVIRI.IR_DIMENSION, MVIRI.IR_DIMENSION, np.float32, fill_value=np.NaN)
        variable = Variable([MVIRI.IR_X_DIMENSION, MVIRI.IR_X_DIMENSION], default_array)
        variable.attrs["standard_name"] = "longitude"
        TemplateUtil.add_units(variable, "degrees_east")
        TemplateUtil.add_encoding(variable, np.int16, -32768, scale_factor=0.0054933317)
        dataset["longitude_ir_wv"] = variable

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        pass    # not required in this class

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        pass    # not required in this class

    @staticmethod
    def get_swath_width():
        return MVIRI.FULL_DIMENSION