from fiduceo.common.writer.templates.templateutil import TemplateUtil as tu

CHUNKING = (1280, 409)


class AOT:

    @staticmethod
    def add_variables(ds, width, height):
        tu.add_geolocation_variables(ds, width, height, chunksizes=CHUNKING)
        tu.add_quality_flags(ds, width, height, chunksizes=CHUNKING)
