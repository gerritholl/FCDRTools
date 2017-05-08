from fiduceo.fcdr.writer.templates.hirs import HIRS


class HIRS3:
    @staticmethod
    def add_original_variables(dataset, height):
        HIRS.add_geolocation_variables(dataset, height)

        HIRS.add_bt_variable(dataset, height)
        HIRS.add_common_angles(dataset, height)

        HIRS.add_original_variables(dataset, height)

    @staticmethod
    def add_easy_fcdr_variables(dataset, height):
        HIRS.add_easy_fcdr_variables(dataset, height)

    @staticmethod
    def add_full_fcdr_variables(dataset, height):
        HIRS.add_full_fcdr_variables(dataset, height)

    @staticmethod
    def get_swath_width():
        return HIRS.get_swath_width()