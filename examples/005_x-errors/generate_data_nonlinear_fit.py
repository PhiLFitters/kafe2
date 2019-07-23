import numpy as np
from kafe2 import XYContainer
from kafe2.fit.util.function_library import linear_model

num_datapoints = 16
a_0 = 1.0
b_0 = 0.0
x_err = 1.0
y_err = 0.1
x_0 = np.arange(num_datapoints)
x_jitter = np.random.normal(loc=0, scale=x_err, size=num_datapoints)
y_0 = linear_model(x_0 + x_jitter, a_0, b_0)
y_jitter = np.random.normal(loc=0, scale=y_err, size=num_datapoints)
y_data = y_0 + y_jitter

data_container = XYContainer(x_data=x_0, y_data=y_data)
data_container.to_file('data.yml')
