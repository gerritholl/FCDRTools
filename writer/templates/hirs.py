from writer.templates.templateutil import TemplateUtil


class HIRS:
    @staticmethod
    def add_original_variables(dataset, height, width):
        TemplateUtil.add_geolocation_variables(dataset, width, height)
