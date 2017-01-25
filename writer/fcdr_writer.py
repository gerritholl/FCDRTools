import xarray as xr

from writer.templates.easy_fcdr import EasyFCDR
from writer.templates.template_factory import TemplateFactory


class FCDRWriter:
    @classmethod
    def write(cls, ds, file, format=None):
        """
        Save a dataset to NetCDF file.
        :param ds: The dataset
        :param file: File path
        :param format: NetCDF format flavour, one of 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC'.
         """
        ds.to_netcdf(file, format=format)

    @classmethod
    def createTemplateEasy(cls, sensorType, height):
        """
        Create a template dataset in EASY FCDR format for the sensor given as argument.
        :param sensorType: the sensor type to create the template for
        :return the template dataset
         """
        dataset = xr.Dataset()
        cls._add_standard_global_attributes(dataset)

        template_factory = TemplateFactory()

        sensor_template = template_factory.get_sensor_template(sensorType)
        sensor_template.add_original_variables(dataset, height)

        EasyFCDR.add_variables(dataset, sensor_template.get_swath_width(), height)

        return dataset

    @classmethod
    def createTemplateFull(cls, sensorType, height):
        """
        Create a template dataset in FULL FCDR format for the sensor given as argument.
        :param sensorType: the sensor type to create the template for
        :return the template dataset
         """
        dataset = xr.Dataset()

        cls._add_standard_global_attributes(dataset)

        template_factory = TemplateFactory()

        sensor_template = template_factory.get_sensor_template(sensorType)
        sensor_template.add_original_variables(dataset, height)
        sensor_template.add_uncertainty_variables(dataset, height)

        return dataset

    @classmethod
    def _add_standard_global_attributes(cls, dataset):
        dataset.attrs["Conventions"] = "CF-1.6"
        dataset.attrs["license"] = "This dataset is released for use under CC-BY licence and was developed in the EC " \
                                   "FIDUCEO project \"Fidelity and Uncertainty in Climate Data Records from Earth " \
                                   "Observations\". Grant Agreement: 638822."
        # @todo tb/tb 2 the following dictionary entries have to be supplied by the data generators
        dataset.attrs["institution"] = None
        dataset.attrs["title"] = None
        dataset.attrs["source"] = None
        dataset.attrs["history"] = None
        dataset.attrs["references"] = None
        dataset.attrs["comment"] = None
