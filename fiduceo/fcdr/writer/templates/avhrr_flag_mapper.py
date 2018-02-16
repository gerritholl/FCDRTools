import numpy as np

from fiduceo.fcdr.writer.global_flags import GlobalFlags as gf


class AVHRR_FlagMapper:
    BAD_GEOLOCATION_TIMING_ERR = np.uint8(1)
    BAD_CALIBRATION_RADIOMETER_ERR = np.uint8(2)

    source_masks = [BAD_GEOLOCATION_TIMING_ERR, BAD_CALIBRATION_RADIOMETER_ERR]
    target_masks = [gf.USE_WITH_CAUTION, gf.USE_WITH_CAUTION]

    def map_global_flags(self, dataset):
        global_flag_data = dataset["quality_pixel_bitmask"].data
        avhrr_flag_data = dataset["data_quality_bitmask"].data

        for source_mask, target_mask in zip(self.source_masks, self.target_masks):
            intermediate = np.logical_and(avhrr_flag_data, source_mask).astype(np.uint8) * target_mask
            global_flag_data = np.bitwise_or(global_flag_data, intermediate)

        dataset["quality_pixel_bitmask"].data = global_flag_data
