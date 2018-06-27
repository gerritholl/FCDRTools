import unittest

from fiduceo.cdr.writer.cdr_writer import CDRWriter
from fiduceo.common.test.assertions import Assertions


class CDRWriterTest(unittest.TestCase):

    def testTemplateEasy_ALBEDO(self):
        ds = CDRWriter.createTemplate('ALBEDO', 5000, 5000)
        self.assertIsNotNone(ds)

        Assertions.assert_global_attributes(self, ds.attrs)
        Assertions.assert_cdr_global_attributes(self, ds.attrs)

    def testTemplateEasy_AOT(self):
        ds = CDRWriter.createTemplate('AOT', 409, 12941)
        self.assertIsNotNone(ds)

        Assertions.assert_global_attributes(self, ds.attrs)
        Assertions.assert_cdr_global_attributes(self, ds.attrs)

    def testTemplateEasy_SST(self):
        ds = CDRWriter.createTemplate('SST', 409, 11732)
        self.assertIsNotNone(ds)

        Assertions.assert_global_attributes(self, ds.attrs)
        Assertions.assert_cdr_global_attributes(self, ds.attrs)

    def testTemplateEasy_SST_ENSEMBLE(self):
        ds = CDRWriter.createTemplate('SST_ENSEMBLE', 409, 11732, 20)
        self.assertIsNotNone(ds)

        Assertions.assert_global_attributes(self, ds.attrs)
        Assertions.assert_cdr_global_attributes(self, ds.attrs)

    def testTemplateEasy_UTH(self):
        ds = CDRWriter.createTemplate('UTH', 360, 120)
        self.assertIsNotNone(ds)

        Assertions.assert_global_attributes(self, ds.attrs)
        Assertions.assert_cdr_global_attributes(self, ds.attrs)
        Assertions.assert_gridded_global_attributes(self, ds.attrs)