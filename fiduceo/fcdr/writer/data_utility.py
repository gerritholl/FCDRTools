import numpy as np


class DataUtility:
    @staticmethod
    def check_scaling_ranges(variable):
        valid_range = DataUtility._get_min_max(variable)

        data_min = np.nanmin(variable.data)
        data_max = np.nanmax(variable.data)

        scale_factor = DataUtility._get_scale_factor(variable)
        add_offset = DataUtility._get_add_offset(variable)

        scaled_min = round((data_min - add_offset) / scale_factor)
        scaled_max = round((data_max - add_offset) / scale_factor)

        if scaled_min < valid_range.min:
            raise ValueError('data scaling underflow: ', data_min)

        if scaled_max > valid_range.max:
            raise ValueError('data scaling overflow: ', data_max)

    @staticmethod
    def _get_scale_factor(variable):
        scale_factor = variable.encoding.get('scale_factor')
        if scale_factor is not None:
            return scale_factor

        return 1.0

    @staticmethod
    def _get_add_offset(variable):
        add_offset = variable.encoding.get('add_offset')
        if add_offset is not None:
            return add_offset

        return 0.0
    
    @staticmethod
    def _get_min_max(variable):
        data_type = variable.encoding.get('dtype')
        if data_type is None:
            raise ValueError('data type missing')

        return np.iinfo(data_type)
