import unittest

from writer.templates.template_factory import TemplateFactory


class TemplateFactoryTest(unittest.TestCase):
    def test_get_template(self):
        template_factory = TemplateFactory()

        amsub = template_factory.get_template("AMSUB")
        self.assertIsNotNone(amsub)

        avhrr = template_factory.get_template("AVHRR")
        self.assertIsNotNone(avhrr)

        hirs = template_factory.get_template("HIRS")
        self.assertIsNotNone(hirs)

        mviri = template_factory.get_template("MVIRI")
        self.assertIsNotNone(mviri)