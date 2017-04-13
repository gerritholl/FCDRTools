from fiduceo.fcdr.writer.templates.avhrr import AVHRR
from fiduceo.fcdr.writer.templates.hirs import HIRS
from fiduceo.fcdr.writer.templates.mviri import MVIRI
from fiduceo.fcdr.writer.templates.amsub_mhs import AMSUB_MHS


class TemplateFactory:
    def __init__(self):
        self.templates = dict([("AMSUB", AMSUB_MHS), ("MHS", AMSUB_MHS), ("AVHRR", AVHRR), ("HIRS", HIRS), ("MVIRI", MVIRI)])

    def get_sensor_template(self, name):
        return self.templates[name]
