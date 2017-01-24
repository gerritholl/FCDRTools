from writer.templates.amsub import AMSUB
from writer.templates.avhrr import AVHRR
from writer.templates.hirs import HIRS
from writer.templates.mviri import MVIRI


class TemplateFactory:

    def __init__(self):
        self.templates = dict([("AMSUB", AMSUB), ("AVHRR", AVHRR), ("HIRS", HIRS), ("MVIRI", MVIRI)])

    def get_sensor_template(self, name):
        return self.templates[name]