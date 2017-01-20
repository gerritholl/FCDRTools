import xarray as xr

from writer.templates.amsub import AMSUB
from writer.templates.avhrr import AVHRR


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
    def createTemplateEasy(cls, sensorType, width, height):
        """
        Create a template dataset for the sensor given as argument.
        :param sensorType: the sensor type to create the template for
        :return the template dataset
         """
        dataset = xr.Dataset()
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

        if sensorType == 'AVHRR':
            AVHRR.add_original_variables(dataset, width, height)
        else:
            AMSUB.add_original_variables(dataset, width, height)

        return dataset
