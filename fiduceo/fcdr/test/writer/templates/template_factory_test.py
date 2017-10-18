import unittest

from fiduceo.fcdr.writer.templates.template_factory import TemplateFactory


class TemplateFactoryTest(unittest.TestCase):
    def test_get_sensor_template(self):
        template_factory = TemplateFactory()

        amsub = template_factory.get_sensor_template("AMSUB")
        self.assertIsNotNone(amsub)

        mhs = template_factory.get_sensor_template("MHS")
        self.assertIsNotNone(mhs)

        ssmts = template_factory.get_sensor_template("SSMT2")
        self.assertIsNotNone(ssmts)

        avhrr = template_factory.get_sensor_template("AVHRR")
        self.assertIsNotNone(avhrr)

        hirs_2 = template_factory.get_sensor_template("HIRS2")
        self.assertIsNotNone(hirs_2)

        hirs_3 = template_factory.get_sensor_template("HIRS3")
        self.assertIsNotNone(hirs_3)

        hirs_4 = template_factory.get_sensor_template("HIRS4")
        self.assertIsNotNone(hirs_4)

        mviri = template_factory.get_sensor_template("MVIRI")
        self.assertIsNotNone(mviri)

        mviri_static = template_factory.get_sensor_template("MVIRI_STATIC")
        self.assertIsNotNone(mviri_static)
