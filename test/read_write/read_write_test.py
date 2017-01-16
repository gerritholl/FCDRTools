import os
import tempfile
import unittest

import xarray as xr

from writer.fcdr_writer import FCDRWriter


class ReadWriteTests(unittest.TestCase):

    def setUp(self):
        tempDir = tempfile.gettempdir()
        self.testDir = os.path.join(tempDir, 'fcdr')
        os.mkdir(self.testDir)

    def tearDown(self):
        if (os.path.isdir(self.testDir)):
            for i in os.listdir(self.testDir):
                os.remove(os.path.join(self.testDir, i))
            os.rmdir(self.testDir)

    def test_write_empty(self):
        testFile = os.path.join(self.testDir, 'delete_me.nc')
        emptyDataset = xr.Dataset()

        FCDRWriter.write(emptyDataset, testFile)

        self.assertTrue(os.path.isfile(testFile))
