import numpy as np
from xarray import Variable

from writer.default_data import DefaultData


class EasyFCDR:
    @staticmethod
    def add_variables(dataset, width, height):
        # u_random
        default_array = DefaultData.create_default_array(width, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "random uncertainty"
        dataset["u_random"] = variable

        # u_systematic
        default_array = DefaultData.create_default_array(width, height, np.float32)
        variable = Variable(["y", "x"], default_array)
        variable.attrs["_FillValue"] = DefaultData.get_default_fill_value(np.float32)
        variable.attrs["standard_name"] = "systematic uncertainty"
        dataset["u_systematic"] = variable
