#!/usr/bin/env python3

"""
This file is a helper for qr_vs_cholesky_speed.py.
Due to garbage collection each fit in the aforementioned benchmark needs to be run in a separate process.
"""

import sys
from time import time
import numpy as np
from kafe2 import XYFit, plot


X_ERR_UNCOR = X_ERR_COR = Y_ERR_UNCOR = Y_ERR_COR = 0.1
a_0 = 1.2
b_0 = 3.4


def benchmark(cost_function, num_points):
   x_data_0 = np.linspace(-10, 10, num_points)
   x_data = x_data_0
   x_data += np.random.normal(scale=X_ERR_UNCOR, size=num_points)
   x_data += np.random.normal(scale=X_ERR_COR, size=1)

   y_data = a_0*x_data + b_0
   y_data += np.random.normal(scale=Y_ERR_UNCOR, size=num_points)
   y_data += np.random.normal(scale=Y_ERR_COR, size=1)

   fit = XYFit([x_data_0, y_data], cost_function=cost_function)
   fit.add_error("x", X_ERR_UNCOR, correlation=0)
   fit.add_error("x", X_ERR_COR, correlation=1)
   fit.add_error("y", Y_ERR_UNCOR, correlation=0)
   fit.add_error("y", Y_ERR_COR, correlation=1)

   fit.do_fit()


if __name__ == "__main__":
    benchmark(sys.argv[1], int(sys.argv[2]))
