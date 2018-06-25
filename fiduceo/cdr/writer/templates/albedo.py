from fiduceo.common.writer.templates.templateutil import TemplateUtil as tu

CHUNKING = (500, 500)

class Albedo:

    @staticmethod
    def add_variables(ds, width, height):
        # @todo 1 tb/tb add geolocation 2018-06-25

        tu.add_quality_flags(ds, width, height, chunksizes=CHUNKING)
