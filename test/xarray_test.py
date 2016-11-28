

import numpy as np
import pandas as pd
import xarray as xr

data = xr.DataArray(np.random.randn(2, 3), [('x', ['a', 'b']), ('y', [-2, 0, 2])])

print(data.values)

subset = data.isel(y=slice(1))

print(subset.values)