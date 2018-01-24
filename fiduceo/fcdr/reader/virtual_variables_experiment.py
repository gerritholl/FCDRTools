import numpy as np
import xarray as xr
from xarray import Variable

from fiduceo.fcdr.reader.nc_temp import nc_file_temp
from fiduceo.fcdr.writer.fcdr_writer import FCDRWriter


def main():
    dataset = xr.Dataset()
    v_var = Variable(["virtual"], np.full([0], np.NaN))
    v_var.attrs["virtual"] = "true"
    # variable.attrs["dimension"] = "[channel height width]"
    v_var.attrs["expression"] = "das * kann + ich"
    dataset["v_var"] = v_var

    shape = (30000, 5000)

    dataset["das"] = Variable(["y", "x"], create_nd_random_data_array(shape, 10, True))
    dataset["kann"] = Variable(["y", "x"], create_nd_random_data_array(shape, 10, True))
    dataset["ich"] = Variable(["y", "x"], create_nd_random_data_array(shape, 1000, True))

    writer = FCDRWriter()
    writer.write(dataset, nc_file_temp, overwrite=True)



def create_nd_random_data_array(shape, multiplier, should_be_integer):
    data_size = get_data_size(shape)
    np_random = np.random.random_sample(data_size) * multiplier
    np_random = np_random.reshape(shape)
    if should_be_integer:
        np_random = np.round(np_random, 0)
        np_random = np.asarray(np_random, int)
    return np_random


def get_data_size(shape):
    shape_size = len(shape)
    data_size = shape[0]
    for i in range(shape_size - 1):
        data_size *= shape[i + 1]
    return data_size


if __name__ == "__main__":
    main()
