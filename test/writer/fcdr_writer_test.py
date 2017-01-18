import unittest
import numpy as np

from writer.fcdr_writer import FCDRWriter


class FCDRWriterTest(unittest.TestCase):
    def testCreateTemplateEasy_AVHRR(self):
        ds = FCDRWriter.createTemplateEasy('AVHRR', 409, 12198)
        self.assertIsNotNone(ds)

        self.verifyGlobalAttributes(ds.attrs)

        latitude = ds.variables["lat"]
        longitude = ds.variables["lon"]

        lonData = np.zeros((409, 12198))
        longitude.data = lonData

    def testCreateTemplateEasy_AMSUB(self):
        ds = FCDRWriter.createTemplateEasy('AMSUB', 90, 2561)
        self.assertIsNotNone(ds)

        self.verifyGlobalAttributes(ds.attrs)

        #latitude = ds.variables["Latitude"]
        #longitude = ds.variables["Longitude"]

    def verifyGlobalAttributes(self, attributes):
        self.assertIsNotNone(attributes)
        self.assertEquals("CF-1.6", attributes["Conventions"])
        self.assertEquals("This dataset is released for use under CC-BY licence and was developed in the EC FIDUCEO "
                          "project “Fidelity and Uncertainty in Climate Data Records from Earth Observations”. Grant "
                          "Agreement: 638822.", attributes["license"])
