from distutils.core import setup

from setuptools import find_packages

from fiduceo.fcdr.writer.fcdr_writer import FCDRWriter

setup(name='fcdr_tools', version=FCDRWriter._version, description='FIDUCEO CDR/FCDR read and write utilities', author='Tom Block', author_email='tom.block@brockmann-consult.de',
      url='http://www.fiduceo.eu', packages=find_packages(), install_requires=['numpy >=1.11.0', 'xarray >=0.8.2', 'netcdf4 >=1.2.4', 'numexpr >=2.6.2', 'dask >= 0.15.2'])
