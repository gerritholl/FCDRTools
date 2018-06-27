import os

import xarray as xr

from fiduceo.cdr.writer.templates.cdr_template_factory import CDR_TemplateFactory
from fiduceo.common.writer.writer_utils import WriterUtils


class CDRWriter:

    @staticmethod
    def write(ds, file, compression_level=None, overwrite=False):
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

        # set up compression parameter for ALL variables. Unfortunately, xarray does not allow
        # one set of compression params per file, only per variable. tb 2018-06-27
        if compression_level is None:
            compression_level = 5

        comp = dict(zlib=True, complevel=compression_level)
        encoding = dict()
        for var_name in ds.data_vars:
            var_encoding = dict(comp)
            var_encoding.update(ds[var_name].encoding)
            encoding.update({var_name: var_encoding})

        ds.to_netcdf(file, format='netCDF4', engine='netcdf4', encoding=encoding)

    @staticmethod
    def createTemplate(data_type, width, height, num_samples=None):
        """
        Create a template dataset in CDR format for the data type given as argument.
        :param data_type: the data type to create the template for
        :param width: the width in pixels of the data product
        :param height: the height in pixels of the data product
        :return the template dataset
         """
        dataset = xr.Dataset()
        WriterUtils.add_standard_global_attributes(dataset)
        WriterUtils.add_cdr_global_attributes(dataset)

        template_factory = CDR_TemplateFactory()

        sensor_template = template_factory.get_cdr_template(data_type)

        if num_samples is None:
            sensor_template.add_variables(dataset, width, height)
        else:
            sensor_template.add_variables(dataset, width, height, num_samples)

        return dataset
