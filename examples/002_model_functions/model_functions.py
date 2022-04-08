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

from kafe2 import XYContainer, Fit, Plot, ContoursProfiler
import numpy as np


# To define a model function for kafe2 simply write it as a Python function.
# Important: The first argument of the model function is interpreted as the independent variable
#     of the fit. It is not being modified during the fit and it's the quantity represented by
#     the x axis of our fit.

# Our first model is a simple linear function:
def linear_model(x, a, b):
    return a * x + b


# Our second model is a simple exponential function.
# The kwargs in the function header specify parameter defaults.
def exponential_model(x, A_0=1., x_0=5.):
    return A_0 * np.exp(x/x_0)


# Read in the measurement data from a yaml file.
# For more information on reading/writing kafe2 objects from/to files see examples 005_convenience
xy_data = XYContainer.from_file("data.yml")

# Create 2 Fit objects with the same data but with different model functions:
linear_fit = Fit(data=xy_data, model_function=linear_model)
exponential_fit = Fit(data=xy_data, model_function=exponential_model)

# Optional: Assign LaTeX strings to parameters and model functions.
# linear_fit.assign_parameter_latex_names(x="X", a=r"\alpha", b=r"\beta")  # Uncomment to activate
linear_fit.assign_model_function_latex_expression("{a}{x} + {b}")
# exponential_fit.assign_parameter_latex_names(x="X", A_0="B_0", x_0="X_0")  # Uncomment to activate
exponential_fit.assign_model_function_latex_expression("{A_0} e^{{{x}/{x_0}}}")

# Perform the fits:
linear_fit.do_fit()
exponential_fit.do_fit()

# Optional: Print out a report on the result of each fit.
linear_fit.report()
exponential_fit.report()

# Optional: Create a plot of the fit results using Plot.
p = Plot(fit_objects=[linear_fit, exponential_fit], separate_figures=False)

# Optional: Customize the plot appearance; only show the data points once.
p.customize('data', 'color', values=['k', 'none'])  # Hide points for second fit.
p.customize('data', 'label', values=['data points', None])  # No second legend entry.

# Do the plotting:
p.plot(fit_info=True)

# Optional: Create a contour plot for the exponential fit to show the parameter correlations.
cpf = ContoursProfiler(exponential_fit)
cpf.plot_profiles_contours_matrix(show_grid_for='contours')

# Show the fit results:
p.show()
