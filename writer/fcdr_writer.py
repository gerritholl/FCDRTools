
import xarray as xr

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
    def createTemplate(cls, sensorType):
        """
        Create a template dataset for the sensor given as argument.
        :param sensorType: the sensor type to create the template for
        :return the template dataset
         """
        dataset = xr.Dataset()
        dataset.attrs["Conventions"] = "CF-1.6"
        dataset.attrs["institution"] = "FIDUCEO" # @todo 2 tb/tb this must be a parameter 2017-01-16
        dataset.attrs["title"] = "FIDUCEO dataset" # @todo 2 tb/tb this must be a parameter 2017-01-16
        dataset.attrs["source"] = "The original data reference" # @todo 2 tb/tb this must be a parameter 2017-01-16
        dataset.attrs["history"] = "What we did" # @todo 2 tb/tb this must be a parameter 2017-01-16
        dataset.attrs["references"] = "http://www.fiduceo.eu/publications" # @todo 2 tb/tb this must be a parameter (?) 2017-01-16
        dataset.attrs["comment"] = "The legal things?" # @todo 2 tb/tb this must be a parameter (?) 2017-01-16
        dataset.attrs["license"] = "The license text" # @todo 2 tb/tb update with real text 2017-01-16

        return dataset