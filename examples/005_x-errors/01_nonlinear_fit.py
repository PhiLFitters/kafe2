#!/usr/bin/env python
"""
kafe2 example: Nonlinear fit
============================

kafe2 fits support the addition of x data errors - in fact we've been using them since the very first example. To take
them into account the x errors are converted to y errors via multiplication with the derivative of the model function.
In other words, kafe2 fits extrapolate the derivative of the model function at the x data values and calculate how a
difference in the x direction would translate to the y direction. Unfortunately this approach is not perfect though.
Since we're extrapolating the derivative at the x data values, we will only receive valid results if the derivative
doesn't change too much at the scale of the x error. Also, since the effective y error has now become dependent on the
derivative of the model function it will vary depending on our choice of model parameters. This distorts our likelihood
function - the minimum of a chi2 cost function will no longer be shaped like a parabola (with a model parameter on the x
axis and chi2 on the y axis). Now, you might be wondering why you should care about the shape of some likelihood
function. The reason why it's important is that the common notation of par_value+-par_error for fit results is only
valid for a parabola-shaped cost function. If your likelihood function is distorted it will also affect your fit
results! A fit with a parabola-shaped cost function is called a linear fit, whereas a fit with a cost function of any
other shape is called a nonlinear fit. To be clear, linear/nonlinear does NOT refer to the model function. You can use
a parabola as your model function and still have a linear fit.

Luckily nonlinear fits oftentimes still produce meaningful fit results as long as the distortion is not too big - you
just need to be more careful during the evaluation of your fit results. A common approach for handling nonlinearity is
to trace the profile of the cost function in either direction of the cost function minimum and find the points at which
the cost function value has increased by a specified amount relative to the cost function minimum. In other words, two
cuts are made on either side of the cost function minimum at a specified height. The two points found with this approach
span a confidence interval for the fit parameter around the cost function minimum. The confidence level of the interval
depends on how high you set the cuts for the cost increase relative to the cost function minimum. The one sigma interval
described by conventional parameter errors is achieved by a cut at the fit minimum plus 1^2=1 and has a confidence level
of about 68%. The two sigma interval is achieved by a cut at the fit minimum plus 2^2=4 and has a confidence level of
about 95%, and so on. The one sigma interval is commonly described by what is called asymmetric errors: the interval
limits are described relative to the cost function minimum as par_value+par_err_up-par_err_down.

In this example we will construct a fit with a nonlinear fit, calculate the asymmetric parameter errors, and compare
them with the conventional parameter errors.
"""

import matplotlib.pyplot as plt
from kafe2 import XYContainer, XYFit, Plot
from kafe2.fit.tools import ContoursProfiler

# Construct a fit with data loaded from a yaml file. The model function is the default of f(x) = a * x + b
nonlinear_fit = XYFit(xy_data=XYContainer.from_file('data.yml'))

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
plot.plot(with_fit_info=True, with_asymmetric_parameter_errors=True)

# Optional: Calculate a detailed representation of the profile likelihood
# Note how the actual chi2 profile differs from the parabolic approximation that you would expect with a linear fit.
profiler = ContoursProfiler(nonlinear_fit)
profiler.plot_profiles_contours_matrix(show_grid_for='all')

plt.show()

