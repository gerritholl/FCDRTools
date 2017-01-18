import unittest

from writer.fcdr_writer import FCDRWriter


class FCDRWriterTest(unittest.TestCase):
    def testCreateTemplateEasy_AVHRR(self):
        ds = FCDRWriter.createTemplateEasy('AVHRR11')

        self.assertIsNotNone(ds)

        attributes = ds.attrs
        self.assertIsNotNone(attributes)

        self.assertEquals("CF-1.6", attributes["Conventions"])
        self.assertEquals("FIDUCEO", attributes["institution"])
        self.assertEquals("FIDUCEO dataset", attributes["title"])
        self.assertEquals("The original data reference", attributes["source"])
        self.assertEquals("What we did", attributes["history"])
        self.assertEquals("http://www.fiduceo.eu/publications", attributes["references"])
        self.assertEquals("The legal things?", attributes["comment"])
        self.assertEquals("This dataset is released for use under CC-BY licence and was developed in the EC FIDUCEO "
                          "project “Fidelity and Uncertainty in Climate Data Records from Earth Observations”. Grant "
                          "Agreement: 638822.", attributes["license"])

        latitude = ds.variables["lat"]
        longitude = ds.variables["lon"]
