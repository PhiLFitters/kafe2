#!/usr/bin/env python
"""
kafe2 example: Model Functions
==============================

In experimental physics a line fit will only suffice for a small number
of applications. In most cases you will need a more complex model function
with more parameters to accurately model physical reality.
This example demonstrates how to specify arbitrary model functions for
a kafe2 fit.
"""

from kafe2 import XYContainer, XYFit, Plot, ContoursProfiler
import numpy as np
import matplotlib.pyplot as plt

# To define a model function for kafe2 simply write it as a python function
# Important: The model function must have the x among its arguments
def linear_model(x, a, b):
    # Our first model is a simple linear function
    return a * x + b

def exponential_model(x, A0=1., x0=5.):
    # Our second model is a simple exponential function
    # The kwargs in the function header specify parameter defaults.
    return A0 * np.exp(x/x0)


# Read in the measurement data from a yaml file.
# For more information on reading/writing kafe2 objects from/to files see TODO
xy_data = XYContainer.from_file("data.yml")

# Create 2 XYFit objects with the same data but with different model functions
linear_fit = XYFit(xy_data=xy_data, model_function=linear_model)
exponential_fit = XYFit(xy_data=xy_data, model_function=exponential_model)

# Optional: Assign LaTeX strings to parameters and model functions.
linear_fit.assign_parameter_latex_names(a='a', b='b')
linear_fit.assign_model_function_latex_expression("{a}{x} + {b}")
exponential_fit.assign_parameter_latex_names(A0='A_0', x0='x_0')
exponential_fit.assign_model_function_latex_expression("{A0} e^{{{x}/{x0}}}")

# Perform the fits.
linear_fit.do_fit()
exponential_fit.do_fit()

# Optional: Print out a report on the result of each fit.
linear_fit.report()
exponential_fit.report()

# Optional: Create a plot of the fit results using Plot.
p = Plot(fit_objects=[linear_fit, exponential_fit], separate_figures=False)
p.plot(with_fit_info=True)

# Optional: Create a contour plot for the exponential fit to show the parameter correlations.
cpf = ContoursProfiler(exponential_fit)
cpf.plot_profiles_contours_matrix(show_grid_for='contours')

# Show the fit results.
plt.show()
