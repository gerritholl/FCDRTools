import unittest

from fiduceo.fcdr.reader.fcdr_reader import FCDRReader
from fiduceo.fcdr.writer.fcdr_writer import FCDRWriter


class FCDRReaderMviriVirtualVariablesTest(unittest.TestCase):

    def testCalculate_sensitivitiy_solar_irradiance(self):
        fcdr_writer = FCDRWriter()
        dataset = fcdr_writer.createTemplateFull("MVIRI", 5000)

        fcdr_reader = FCDRReader()
        # @todo 1 tb/tb continue here 2018-02-23
        # variable = fcdr_reader._load_virtual_variable(dataset, "sensitivity_solar_irradiance_vis")
        # self.assertIsNotNone(variable)