from writer.templates.amsub_mhs import AMSUB_MHS
from writer.templates.avhrr import AVHRR
from writer.templates.hirs import HIRS
from writer.templates.mviri import MVIRI


class TemplateFactory:
    def __init__(self):
        self.templates = dict([("AMSUB", AMSUB_MHS), ("MHS", AMSUB_MHS), ("AVHRR", AVHRR), ("HIRS", HIRS), ("MVIRI", MVIRI)])

    def get_sensor_template(self, name):
        return self.templates[name]
