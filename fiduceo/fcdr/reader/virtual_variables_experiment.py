import numpy as np
import numexpr as ne
import xarray as xr
from xarray import Variable

from fcdr_writer import FCDRWriter


def main():
    # a = np.ones((100, 100))
    # b = np.ones((100, 100))
    # d = np.ones((100, 100)) * 3.1415927
    #
    # c = ne.evaluate("sin(d)")
    #
    # print(c)

    dataset = xr.Dataset()
    variable = Variable(["virtual"], np.full([0], np.NaN))
    variable.attrs["virtual"] = "true"
    variable.attrs["dimension"] = "[channel height width]"
    variable.attrs["expression"] = "alpha * Beta + Lametta"
    dataset["Heinz"] = variable

    writer = FCDRWriter()
    writer.write(dataset, "D:\\Satellite\\DELETE\\test.nc", overwrite=True)


if __name__ == "__main__":
    main()