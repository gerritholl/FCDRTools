from writer.templates.templateutil import TemplateUtil

SWATH_WIDTH = 409

class AVHRR:
    @staticmethod
    def add_original_variables(dataset, height):
        TemplateUtil.add_geolocation_variables(dataset, SWATH_WIDTH, height)
