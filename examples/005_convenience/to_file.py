"""
kafe2 example: Saving Fits
==========================

Most kafe2 objects can be turned into the human-readable YAML format and written to a file.
These files can then be used to load the objects into Python code or as input for kafe2go.
"""

from kafe2 import XYContainer, XYFit, Fit, Plot
import matplotlib.pyplot as plt

# The same data as in 001_line_fit/line_fit.py :
xy_data = XYContainer(x_data=[1.0, 2.0, 3.0, 4.0],
                      y_data=[2.3, 4.2, 7.5, 9.4])
xy_data.add_error(axis='x', err_val=0.1)
xy_data.add_error(axis='y', err_val=0.4)

# Save the data container to a file:
xy_data.to_file("data_container.yml")
# Because a data container does not have a model function running kafe2go with this file as input
#     will fit a linear model a * x + b .


def quadratic_model(x, a, b, c):
    return a * x ** 2 + b * x + c


line_fit = Fit(data=xy_data, model_function=quadratic_model)

# Save the fit to a file:
line_fit.to_file("fit_before_do_fit.yml")
# Because the fit object contains a model function running kafe2go with this file as input will
#     make use of the model function we defined above.

line_fit.do_fit()

# Save the fit with fit results to a file:
line_fit.to_file("fit_after_do_fit.yml")

# Load it back into code:
loaded_fit = XYFit.from_file("fit_after_do_fit.yml")
# Note: this requires the use of a specific fit class like XYFit. The generic Fit pseudo-class
#     does NOT work.

loaded_fit.report()

plot = Plot(fit_objects=loaded_fit)

plot.plot()
plt.show()
