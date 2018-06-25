from fiduceo.common.writer.templates.templateutil import TemplateUtil as tu

class UTH:

    @staticmethod
    def add_variables(ds, width, height):
        tu.add_gridded_geolocation_variables(ds, width, height)
        tu.add_quality_flags(ds, width, height)