import unittest

from writer.templates.template_factory import TemplateFactory


class TemplateFactoryTest(unittest.TestCase):
    def test_get_sensor_template(self):
        template_factory = TemplateFactory()

        amsub = template_factory.get_sensor_template("AMSUB")
        self.assertIsNotNone(amsub)

        mhs = template_factory.get_sensor_template("MHS")
        self.assertIsNotNone(mhs)

        avhrr = template_factory.get_sensor_template("AVHRR")
        self.assertIsNotNone(avhrr)

        hirs = template_factory.get_sensor_template("HIRS")
        self.assertIsNotNone(hirs)

        mviri = template_factory.get_sensor_template("MVIRI")
        self.assertIsNotNone(mviri)
