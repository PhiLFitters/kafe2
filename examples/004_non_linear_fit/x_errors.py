#!/usr/bin/env python
"""
kafe2 example: x_errors
=======================

kafe2 fits support the addition of x data errors - in fact we've been using them since the very first example. To take
them into account the x errors are converted to y errors via multiplication with the derivative of the model function.
In other words, kafe2 fits extrapolate the derivative of the model function at the x data values and calculate how a
difference in the x direction would translate to the y direction. Unfortunately this approach is not perfect though.
Since we're extrapolating the derivative at the x data values, we will only receive valid results if the derivative
doesn't change too much at the scale of the x error. Also, since the effective y error has now become dependent on the
derivative of the model function it will vary depending on our choice of model parameters. This distorts our likelihood
function - the minimum of a chi2 cost function will no longer be shaped like a parabola (with a model parameter on the x
axis and chi2 on the y axis).

The effects of this deformation are explained in the non_linear_fit.py example.
"""

import matplotlib.pyplot as plt
from kafe2 import XYContainer, XYFit, Plot
from kafe2.fit.tools import ContoursProfiler

# Construct a fit with data loaded from a yaml file. The model function is the default of f(x) = a * x + b
nonlinear_fit = XYFit(xy_data=XYContainer.from_file('x_errors.yml'))

# The x errors are much bigger than the y errors. This will cause a distortion of the likelihood function.
nonlinear_fit.add_simple_error('x', 1.0)
nonlinear_fit.add_simple_error('y', 0.1)

# Perform the fit.
nonlinear_fit.do_fit()

# Optional: Print out a report on the fit results on the console.
# Note the asymmetric_parameter_errors flag
nonlinear_fit.report(asymmetric_parameter_errors=True)

# Optional: Create a plot of the fit results using Plot.
# Note the asymmetric_parameter_errors flag
plot = Plot(nonlinear_fit)
plot.plot(fit_info=True, asymmetric_parameter_errors=True)

# Optional: Calculate a detailed representation of the profile likelihood
# Note how the actual chi2 profile differs from the parabolic approximation that you would expect with a linear fit.
profiler = ContoursProfiler(nonlinear_fit)
profiler.plot_profiles_contours_matrix(show_grid_for='all')

plt.show()
