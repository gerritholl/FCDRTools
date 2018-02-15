import os
import xarray as xr

from fiduceo.fcdr.writer.templates.template_factory import TemplateFactory

DATE_PATTERN = "%Y%m%d%H%M%S"


class FCDRWriter:
    _version = "1.1.2"

    @classmethod
    def write(cls, ds, file, compression_level=None, overwrite=False):
        """
        Save a dataset to NetCDF file.
        :param ds: The dataset
        :param file: File path
        :param compression_level: the file compression level, 0 - 9, default is 5
        :param overwrite: set true to overwrite existing files
         """
        if os.path.isfile(file):
            if overwrite is True:
                os.remove(file)
            else:
                raise IOError("The file already exists: " + file)

        # trigger mapping of sensor specific flags to the global flag variable
        template_factory = TemplateFactory()
        flag_mapper = template_factory.get_flag_mapper(ds.attrs["template_key"])
        flag_mapper.map_global_flags(ds)

        # set up compression parameter for ALL variables. Unfortunately, xarray does not allow
        # one set of compression params per file, only per variable. tb 2017-01-25
        if compression_level is None:
            compression_level = 5

        comp = dict(zlib=True, complevel=compression_level)
        encoding = dict()
        for var_name in ds.data_vars:
            var_encoding = dict(comp)
            var_encoding.update(ds[var_name].encoding)
            encoding.update({var_name : var_encoding})

        ds.to_netcdf(file, format='netCDF4', engine='netcdf4', encoding=encoding)

    @classmethod
    def createTemplateEasy(cls, sensorType, height):
        """
        Create a template dataset in EASY FCDR format for the sensor given as argument.
        :param sensorType: the sensor type to create the template for
        :param height the hheight in pixels of the data product
        :return the template dataset
         """
        dataset = xr.Dataset()
        cls._add_standard_global_attributes(dataset)

        template_factory = TemplateFactory()

        sensor_template = template_factory.get_sensor_template(sensorType)
        sensor_template.add_original_variables(dataset, height)
        sensor_template.add_easy_fcdr_variables(dataset, height)
        sensor_template.add_template_key(dataset)

        return dataset

    @classmethod
    def createTemplateFull(cls, sensorType, height):
        """
        Create a template dataset in FULL FCDR format for the sensor given as argument.
        :param sensorType: the sensor type to create the template for
        :param height the hheight in pixels of the data product
        :return the template dataset
         """
        dataset = xr.Dataset()

        cls._add_standard_global_attributes(dataset)

        template_factory = TemplateFactory()

        sensor_template = template_factory.get_sensor_template(sensorType)
        sensor_template.add_original_variables(dataset, height)
        sensor_template.add_full_fcdr_variables(dataset, height)
        sensor_template.add_template_key(dataset)

        return dataset

    @staticmethod
    def create_file_name_FCDR_easy(sensor, platform, start, end, version):
        """
        Create a file name for EASY FCDR format .
        :param sensor: the sensor name
        :param platform: the name of the satellite platform
        :param start: the acquisition start date and time, type datetime
        :param end: the acquisition end date and time, type datetime
        :param version: the processor version string, format "xx.x"
        :return a valid file name
         """
        return FCDRWriter._create_file_name(end, platform, sensor, start, "EASY", version)

    @staticmethod
    def create_file_name_FCDR_full(sensor, platform, start, end, version):
        """
        Create a file name for FULL FCDR format .
        :param sensor: the sensor name
        :param platform: the name of the satellite platform
        :param start: the acquisition start date and time, type datetime
        :param end: the acquisition end date and time, type datetime
        :param version: the processor version string, format "xx.x"
        :return a valid file name
         """
        return FCDRWriter._create_file_name(end, platform, sensor, start, "FULL", version)

    @classmethod
    def _add_standard_global_attributes(cls, dataset):
        dataset.attrs["Conventions"] = "CF-1.6"
        dataset.attrs[
            "licence"] = "This dataset is released for use under CC-BY licence (https://creativecommons.org/licenses/by/4.0/) and was developed in the EC " \
                         "FIDUCEO project \"Fidelity and Uncertainty in Climate Data Records from Earth " \
                         "Observations\". Grant Agreement: 638822."
        dataset.attrs["writer_version"] = FCDRWriter._version

        # The following dictionary entries have to be supplied by the data generators
        dataset.attrs["institution"] = None
        dataset.attrs["title"] = None
        dataset.attrs["source"] = None
        dataset.attrs["history"] = None
        dataset.attrs["references"] = None
        dataset.attrs["comment"] = None

    @staticmethod
    def _create_file_name(end, platform, sensor, start, type, version):
        start_string = start.strftime(DATE_PATTERN)
        end_string = end.strftime(DATE_PATTERN)
        return "FIDUCEO_FCDR_L1C_" + sensor + "_" + platform + "_" + start_string + "_" + end_string + "_" + type + "_v" + version + "_fv" + FCDRWriter._version + ".nc"
