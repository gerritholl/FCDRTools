from writer.fcdr_writer import FCDRWriter


def main():
    writer = FCDRWriter()

    # get a template for sensor name in EASY format, supply product height
    # The scan-width is set automatically
    dataset = writer.createTemplateEasy("AVHRR", 12835)

    # set some mandatory global attributes. Writing will fail if not all of them are filled
    dataset.attrs["institution"] = "Brockmann Consult GmbH"
    dataset.attrs["title"] = "His Majesty!"
    dataset.attrs["source"] = "fake data"
    dataset.attrs["history"] = "none"
    dataset.attrs["references"] = "CDF_FCDR_File Spec"
    dataset.attrs["comment"] = "just to show ho things are intended to be used"

    # write real data to the variables. All variables initially contain "_FillValue".
    # Not writing to the whole array is completely OK
    dataset.variables["Ch1_Bt"].data[23, 44] = 0.456

    # dump it to disk, netcdf4, medium compression
    # writing will fail when the target file already exists
    writer.write(dataset, "D:\\Satellite\\DELETE\\avhrr_fcdr_easy.nc")


if __name__ == "__main__":
    main()
