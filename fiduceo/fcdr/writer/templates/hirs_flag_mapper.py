import numpy as np

from fiduceo.fcdr.writer.global_flags import GlobalFlags as gf
from fiduceo.fcdr.writer.templates.default_flag_mapper import DefaultFlagMapper


class HIRS_FlagMapper(DefaultFlagMapper):

    UNCERTAINTY_SUSPICIOUS = np.uint16(1)

    source_masks = [UNCERTAINTY_SUSPICIOUS]
    target_masks = [gf.USE_WITH_CAUTION]

    def map_global_flags(self, dataset):
        global_flag_data = dataset["quality_pixel_bitmask"].data
        hirs_flag_data = dataset["data_quality_bitmask"].data

        global_flag_data = self.evaluate_masks_uint8(hirs_flag_data, global_flag_data, self.source_masks, self.target_masks)

        dataset["quality_pixel_bitmask"].data = global_flag_data