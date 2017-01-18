import numpy as np
import xarray as xr
from xarray import Variable


class FCDRWriter:
    @classmethod
    def write(cls, ds: xr.Dataset, file: str, format: str = None):
        """
        Save a dataset to NetCDF file.
        :param ds: The dataset
        :param file: File path
        :param format: NetCDF format flavour, one of 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC'.
         """
        ds.to_netcdf(file, format=format)

    @classmethod
    def createTemplateEasy(cls, sensorType):
        """
        Create a template dataset for the sensor given as argument.
        :param sensorType: the sensor type to create the template for
        :return the template dataset
         """
        dataset = xr.Dataset()
        dataset.attrs["Conventions"] = "CF-1.6"
        dataset.attrs["license"] = "This dataset is released for use under CC-BY licence and was developed in the EC " \
                                   "FIDUCEO project “Fidelity and Uncertainty in Climate Data Records from Earth " \
                                   "Observations”. Grant Agreement: 638822."
        dataset.attrs["institution"] = "FIDUCEO"  # @todo 2 tb/tb this must be a parameter 2017-01-16
        dataset.attrs["title"] = "FIDUCEO dataset"  # @todo 2 tb/tb this must be a parameter 2017-01-16
        dataset.attrs["source"] = "The original data reference"  # @todo 2 tb/tb this must be a parameter 2017-01-16
        dataset.attrs["history"] = "What we did"  # @todo 2 tb/tb this must be a parameter 2017-01-16
        dataset.attrs["references"] = "http://www.fiduceo.eu/publications"  # @todo 2 tb/tb this must be a parameter (?) 2017-01-16
        dataset.attrs["comment"] = "The legal things?"  # @todo 2 tb/tb this must be a parameter (?) 2017-01-16

        defaultArray = xr.DataArray(np.random.rand(2, 2), coords=[[0,1], [0,1]] ,dims=['x', 'y'])
        dataset["lat"] = Variable({"x", "y"}, defaultArray)
        dataset["lon"] = Variable({"x", "y"}, defaultArray)

        return dataset
