"""
Create an XYContainer holding the measurement data and uncertainties and write it to a file.
"""

import numpy as np
from kafe2 import XYContainer


START, STOP, NUM_DATAPOINTS = 0, 20, 7
X_ERROR = 0.3  # Absolute error
Y_ERROR = 0.15  # Relative error

np.random.seed(163678643)  # Seeding makes pseudo-random numbers the same every time.


def exponential_model(x, A0=2, x0=7):
    # Generate simulated data with one of the model functions
    return A0 * np.exp(x/x0)


x_data_0 = (STOP-START)*np.random.rand(NUM_DATAPOINTS)+START
x_data_jitter = np.random.normal(loc=0, scale=X_ERROR, size=NUM_DATAPOINTS)
x_data = x_data_0 + x_data_jitter

y_data_0 = exponential_model(x=x_data_0)
y_data_jitter = np.random.normal(loc=0, scale=Y_ERROR, size=NUM_DATAPOINTS)
y_data = y_data_0 * (1.0 + y_data_jitter)

data = XYContainer(x_data=x_data, y_data=y_data)

data.add_simple_error('x', X_ERROR)
data.add_simple_error('y', Y_ERROR, relative=True)

data.to_file('data.yml')
