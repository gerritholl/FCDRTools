import numpy as np
from xarray import Variable

from writer.default_data import DefaultData
from writer.templates.templateutil import TemplateUtil as tu

SWATH_WIDTH = 409
PRT_WIDTH = 3


class AVHRR:
    @staticmethod
    def add_original_variables(dataset, height):
        tu.add_geolocation_variables(dataset, SWATH_WIDTH, height)

        # Time
        variable = tu.create_float_variable(SWATH_WIDTH, height, "time")
        variable.attrs["long_name"] = "Acquisition time in seconds since 1970-01-01 00:00:00"
        tu.add_units(variable, "s")
        dataset["Time"] = variable

        # scanline
        default_array = DefaultData.create_default_vector(height, np.int16)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Level 1b line number"
        variable.attrs["valid_min"] = 0
        dataset["scanline"] = variable

        # satellite_azimuth_angle
        variable = tu.create_float_variable(SWATH_WIDTH, height, "sensor_azimuth_angle")
        tu.add_units(variable, "degree")
        dataset["satellite_azimuth_angle"] = variable

        # satellite_zenith_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["standard_name"] = "sensor_zenith_angle"
        variable.attrs["add_offset"] = 0.0
        tu.add_scale_factor(variable, 0.01)
        tu.add_units(variable, "degree")
        variable.attrs["valid_max"] = 9000
        variable.attrs["valid_min"] = 0
        dataset["satellite_zenith_angle"] = variable

        # solar_azimuth_angle
        variable = tu.create_float_variable(SWATH_WIDTH, height, "solar_azimuth_angle")
        tu.add_units(variable, "degree")
        dataset["solar_azimuth_angle"] = variable

        # solar_zenith_angle
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["standard_name"] = "solar_zenith_angle"
        variable.attrs["add_offset"] = 0.0
        tu.add_scale_factor(variable, 0.01)
        tu.add_units(variable, "degree")
        variable.attrs["valid_max"] = 18000
        variable.attrs["valid_min"] = 0
        dataset["solar_zenith_angle"] = variable

        # Ch1_Ref
        variable = AVHRR._create_channel_refl_variable(height, "Channel 1 Reflectance")
        dataset["Ch1_Ref"] = variable

        # Ch2_Ref
        variable = AVHRR._create_channel_refl_variable(height, "Channel 2 Reflectance")
        dataset["Ch2_Ref"] = variable

        # Ch3a_Ref
        variable = AVHRR._create_channel_refl_variable(height, "Channel 3a Reflectance")
        dataset["Ch3a_Ref"] = variable

        # Ch3b_Bt
        variable = AVHRR._create_channel_bt_variable(height, "Channel 3b Brightness Temperature")
        dataset["Ch3b_Bt"] = variable

        # Ch4_Bt
        variable = AVHRR._create_channel_bt_variable(height, "Channel 4 Brightness Temperature")
        dataset["Ch4_Bt"] = variable

        # Ch5_Bt
        variable = AVHRR._create_channel_bt_variable(height, "Channel 5 Brightness Temperature")
        dataset["Ch5_Bt"] = variable

        # T_ICT
        default_array = DefaultData.create_default_vector(height, np.int16)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["long_name"] = "Temperature of the internal calibration target"
        AVHRR._add_temperature_attributes(variable)
        dataset["T_ICT"] = variable

    @staticmethod
    def get_swath_width():
        return SWATH_WIDTH

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        # u_random_Ch1-3a
        long_names = ["random uncertainty per pixel for channel 1", "random uncertainty per pixel for channel 2",
                          "random uncertainty per pixel for channel 3a"]
        names = ["u_random_Ch1", "u_random_Ch2", "u_random_Ch3a"]
        AVHRR._add_refl_uncertainties_variables_long_name(dataset, height, names, long_names)

        # u_non_random_Ch1-3a
        long_names = ["non-random uncertainty per pixel for channel 1", "non-random uncertainty per pixel for channel 2",
                      "non-random uncertainty per pixel for channel 3a"]
        names = ["u_non_random_Ch1", "u_non_random_Ch2", "u_non_random_Ch3a"]
        AVHRR._add_refl_uncertainties_variables_long_name(dataset, height, names, long_names)

        # u_random_Ch3b-5
        long_names = ["random uncertainty per pixel for channel 3b", "random uncertainty per pixel for channel 4",
                      "random uncertainty per pixel for channel 5"]
        names = ["u_random_Ch3b", "u_random_Ch4", "u_random_Ch5"]
        AVHRR._add_bt_uncertainties_variables_long_name(dataset, height, names, long_names)

        # u_non_random_Ch3b-5
        long_names = ["non-random uncertainty per pixel for channel 3b", "non-random uncertainty per pixel for channel 4",
                      "non-random uncertainty per pixel for channel 5"]
        names = ["u_non_random_Ch3b", "u_non_random_Ch4", "u_non_random_Ch5"]
        AVHRR._add_bt_uncertainties_variables_long_name(dataset, height, names, long_names)

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        # u_latitude
        variable = AVHRR._create_angle_uncertainty_variable("latitude", height)
        dataset["u_latitude"] = variable

        # u_longitude
        variable = AVHRR._create_angle_uncertainty_variable("longitude", height)
        dataset["u_longitude"] = variable

        # u_time
        variable = tu.create_float_variable(SWATH_WIDTH, height, "uncertainty of time")
        tu.add_units(variable, "s")
        dataset["u_time"] = variable

        # u_satellite_azimuth_angle
        variable = AVHRR._create_angle_uncertainty_variable("satellite azimuth angle", height)
        dataset["u_satellite_azimuth_angle"] = variable

        # u_satellite_zenith_angle
        variable = AVHRR._create_angle_uncertainty_variable("satellite zenith angle", height)
        dataset["u_satellite_zenith_angle"] = variable

        # u_solar_azimuth_angle
        variable = AVHRR._create_angle_uncertainty_variable("solar azimuth angle", height)
        dataset["u_solar_azimuth_angle"] = variable

        # u_solar_zenith_angle
        variable = AVHRR._create_angle_uncertainty_variable("solar zenith angle", height)
        dataset["u_solar_zenith_angle"] = variable

        # PRT_C
        default_array = DefaultData.create_default_array(PRT_WIDTH, height, np.int16)
        variable = Variable(["y", "n_prt"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["standard_name"] = "Prt counts"
        tu.add_units(variable, "count")
        dataset["PRT_C"] = variable

        # u_prt
        default_array = DefaultData.create_default_array(PRT_WIDTH, height, np.float32)
        variable = Variable(["y", "n_prt"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "Uncertainty on the PRT counts"
        tu.add_units(variable, "count")
        dataset["u_prt"] = variable

        # R_ICT
        default_array = DefaultData.create_default_array(PRT_WIDTH, height, np.float32)
        variable = Variable(["y", "n_prt"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "Radiance of the PRT"
        tu.add_units(variable, "mW m^-2 sr^-1 cm")
        dataset["R_ICT"] = variable

        # T_instr
        default_array = DefaultData.create_default_vector(height, np.float32)
        variable = Variable(["y"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.float32))
        variable.attrs["standard_name"] = "Instrument temperature"
        tu.add_units(variable, "K")
        dataset["T_instr"] = variable

        # Chx_Csp
        standard_names = ["Ch1 Space counts", "Ch2 Space counts", "Ch3a Space counts", "Ch3b Space counts",
                          "Ch4 Space counts", "Ch5 Space counts"]
        names = ["Ch1_Csp", "Ch2_Csp", "Ch3a_Csp", "Ch3b_Csp", "Ch4_Csp", "Ch5_Csp"]
        AVHRR._add_counts_variables(dataset, height, names, standard_names)

        # Chx_Cict
        standard_names = ["Ch3b ICT counts", "Ch4 ICT counts", "Ch5 ICT counts"]
        names = ["Ch3b_Cict", "Ch4_Cict", "Ch5_Cict"]
        AVHRR._add_counts_variables(dataset, height, names, standard_names)

        # Chx_Ce
        standard_names = ["Ch1 Earth counts", "Ch2 Earth counts", "Ch3a Earth counts", "Ch3b Earth counts",
                          "Ch4 Earth counts", "Ch5 Earth counts"]
        names = ["Ch1_Ce", "Ch2_Ce", "Ch3a_Ce", "Ch3b_Ce", "Ch4_Ce", "Ch5_Ce"]
        AVHRR._add_counts_variables(dataset, height, names, standard_names)

        # Chx_u_Csp
        standard_names = ["Ch1 Uncertainty on space counts", "Ch2 Uncertainty on space counts",
                          "Ch3a Uncertainty on space counts", "Ch3b Uncertainty on space counts",
                          "Ch4 Uncertainty on space counts", "Ch5 Uncertainty on space counts"]
        names = ["Ch1_u_Csp", "Ch2_u_Csp", "Ch3a_u_Csp", "Ch3b_u_Csp", "Ch4_u_Csp", "Ch5_u_Csp"]
        AVHRR._add_counts_uncertainties_variables(dataset, height, names, standard_names)

        # Chx_Cict
        standard_names = ["Ch3b Uncertainty on ICT counts", "Ch4 Uncertainty on ICT counts",
                          "Ch5 Uncertainty on ICT counts"]
        names = ["Ch3b_u_Cict", "Ch4_u_Cict", "Ch5_u_Cict"]
        AVHRR._add_counts_uncertainties_variables(dataset, height, names, standard_names)

        # Chx_u_Ce
        standard_names = ["Ch1 Uncertainty on earth counts", "Ch2 Uncertainty on earth counts",
                          "Ch3a Uncertainty on earth counts", "Ch3b Uncertainty on earth counts",
                          "Ch4 Uncertainty on earth counts", "Ch5 Uncertainty on earth counts"]
        names = ["Ch1_u_Ce", "Ch2_u_Ce", "Ch3a_u_Ce", "Ch3b_u_Ce", "Ch4_u_Ce", "Ch5_u_Ce"]
        AVHRR._add_counts_uncertainties_variables(dataset, height, names, standard_names)

        # Chx_u_Refl
        standard_names = ["Ch1 Total uncertainty on reflectance", "Ch2 Total uncertainty on reflectance",
                          "Ch3a Total uncertainty on reflectance"]
        names = ["Ch1_u_Refl", "Ch2_u_Refl", "Ch3a_u_Refl"]
        AVHRR._add_refl_uncertainties_variables(dataset, height, names, standard_names)

        # Chx_u_Bt
        standard_names = ["Ch3b Total uncertainty on brightness temperature",
                          "Ch4 Total uncertainty on brightness temperature",
                          "Ch5 Total uncertainty on brightness temperature"]
        names = ["Ch3b_u_Bt", "Ch4_u_Bt", "Ch5_u_Bt"]
        AVHRR._add_bt_uncertainties_variables(dataset, height, names, standard_names)

        # Chx_ur_Bt
        standard_names = ["Ch3b Random uncertainty on brightness temperature",
                          "Ch4 Random uncertainty on brightness temperature",
                          "Ch5 Random uncertainty on brightness temperature"]
        names = ["Ch3b_ur_Bt", "Ch4_ur_Bt", "Ch5_ur_Bt"]
        AVHRR._add_bt_uncertainties_variables(dataset, height, names, standard_names)

        # Chx_us_Bt
        standard_names = ["Ch3b Systematic uncertainty on brightness temperature",
                          "Ch4 Systematic uncertainty on brightness temperature",
                          "Ch5 Systematic uncertainty on brightness temperature"]
        names = ["Ch3b_us_Bt", "Ch4_us_Bt", "Ch5_us_Bt"]
        AVHRR._add_bt_uncertainties_variables(dataset, height, names, standard_names)

    @staticmethod
    def _add_counts_uncertainties_variables(dataset, height, names, standard_names):
        for i, name in enumerate(names):
            variable = AVHRR._create_counts_uncertainty_variable(height, standard_names[i])
            dataset[name] = variable

    @staticmethod
    def _add_bt_uncertainties_variables(dataset, height, names, standard_names):
        for i, name in enumerate(names):
            variable = AVHRR._create_bt_uncertainty_variable(height, standard_name=standard_names[i])
            dataset[name] = variable

    @staticmethod
    def _add_bt_uncertainties_variables_long_name(dataset, height, names, long_names):
        for i, name in enumerate(names):
            variable = AVHRR._create_bt_uncertainty_variable(height, long_name=long_names[i])
            dataset[name] = variable

    @staticmethod
    def _add_refl_uncertainties_variables(dataset, height, names, standard_names):
        for i, name in enumerate(names):
            variable = AVHRR._create_refl_uncertainty_variable(height, standard_name=standard_names[i])
            dataset[name] = variable

    @staticmethod
    def _add_refl_uncertainties_variables_long_name(dataset, height, names, long_names):
        for i, name in enumerate(names):
            variable = AVHRR._create_refl_uncertainty_variable(height, long_name=long_names[i])
            dataset[name] = variable

    @staticmethod
    def _add_counts_variables(dataset, height, names, standard_names):
        for i, name in enumerate(names):
            variable = AVHRR._create_counts_variable(height, standard_names[i])
            dataset[name] = variable

    @staticmethod
    def _create_counts_uncertainty_variable(height, standard_name):
        variable = tu.create_float_variable(SWATH_WIDTH, height, standard_name)
        tu.add_units(variable, "count")
        return variable

    @staticmethod
    def _create_counts_variable(height, standard_name):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int32)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int32))
        variable.attrs["standard_name"] = standard_name
        tu.add_units(variable, "count")
        return variable

    @staticmethod
    def _create_angle_uncertainty_variable(angle_name, height):
        variable = tu.create_float_variable(SWATH_WIDTH, height, "uncertainty of " + angle_name)
        tu.add_units(variable, "degree")
        return variable

    @staticmethod
    def _create_refl_uncertainty_variable(height, standard_name=None, long_name=None):
        variable = tu.create_float_variable(SWATH_WIDTH, height, standard_name=standard_name, long_name=long_name)
        tu.add_units(variable, "percent")
        return variable

    @staticmethod
    def _create_bt_uncertainty_variable(height, standard_name=None, long_name=None):
        variable = tu.create_float_variable(SWATH_WIDTH, height, standard_name=standard_name, long_name=long_name)
        tu.add_units(variable, "K")
        return variable

    @staticmethod
    def _add_temperature_attributes(variable):
        variable.attrs["add_offset"] = 273.15
        tu.add_scale_factor(variable, 0.01)
        tu.add_units(variable, "K")
        variable.attrs["valid_max"] = 10000
        variable.attrs["valid_min"] = -20000

    @staticmethod
    def _create_channel_refl_variable(height, long_name):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["standard_name"] = "toa_reflectance"
        variable.attrs["long_name"] = long_name
        variable.attrs["add_offset"] = 0.0
        tu.add_scale_factor(variable, 1e-4)
        tu.add_units(variable, "percent")
        variable.attrs["valid_max"] = 15000
        variable.attrs["valid_min"] = 0
        return variable

    @staticmethod
    def _create_channel_bt_variable(height, long_name):
        default_array = DefaultData.create_default_array(SWATH_WIDTH, height, np.int16)
        variable = Variable(["y", "x"], default_array)
        tu.add_fill_value(variable, DefaultData.get_default_fill_value(np.int16))
        variable.attrs["standard_name"] = "toa_brightness_temperature"
        variable.attrs["long_name"] = long_name
        AVHRR._add_temperature_attributes(variable)
        return variable
