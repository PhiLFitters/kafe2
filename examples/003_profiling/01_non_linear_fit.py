#!/usr/bin/env python
"""
kafe2 example: non-linear fit
=============================
Very often, when the fit model is a non-linear function of the parameters, the chi2 function is not parabolic around
the minimum. A very common example of such a case is an exponential function parametrized as shown in this example.
In the case of a nonlinear fit, the minimum of a chi2 cost function is not longer shaped like a parabola (with a model
parameter on the x axis and chi2 on the y axis). Now, you might be wondering why you should care about the shape of the
chi2 function. The reason why it's important is that the common notation of par_value+-par_error for fit results is only
valid for a parabola-shaped cost function. If your likelihood function is distorted it will also affect your fit
results!

Luckily nonlinear fits oftentimes still produce meaningful fit results as long as the distortion is not too big - you
just need to be more careful during the evaluation of your fit results. A common approach for handling nonlinearity is
to trace the profile of the cost function (in this case chi2) in either direction of the cost function minimum and find
the points at which the cost function value has increased by a specified amount relative to the cost function minimum.
In other words, two cuts are made on either side of the cost function minimum at a specified height. The two points
found with this approach span a confidence interval for the fit parameter around the cost function minimum. The
confidence level of the interval depends on how high you set the cuts for the cost increase relative to the cost
function minimum. The one sigma interval described by conventional parameter errors is achieved by a cut at the fit
minimum plus 1^2=1 and has a confidence level of about 68%. The two sigma interval is achieved by a cut at the fit
minimum plus 2^2=4 and has a confidence level of about 95%, and so on. The one sigma interval is commonly described by
what is called asymmetric errors: the interval limits are described relative to the cost function minimum as
par_value+par_err_up-par_err_down.
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import Fit, Plot, ContoursProfiler

def exponential(x, A_0=1, tau=1):
    return A_0 * np.exp(-x/tau)

# define the data as simple Python lists
x = [8.018943e-01, 1.839664e+00, 1.941974e+00, 1.276013e+00, 2.839654e+00, 3.488302e+00, 3.775855e+00, 4.555187e+00,
     4.477186e+00, 5.376026e+00]
xerr = 3.000000e-01
y = [2.650644e-01, 1.472682e-01, 8.077234e-02, 1.850181e-01, 5.326301e-02, 1.984233e-02, 1.866309e-02, 1.230001e-02,
     9.694612e-03, 2.412357e-03]
yerr = [1.060258e-01, 5.890727e-02, 3.230893e-02, 7.400725e-02, 2.130520e-02, 7.936930e-03, 7.465238e-03, 4.920005e-03,
        3.877845e-03, 9.649427e-04]

# create a fit object from the data arrays
fit = Fit(data=[x, y], model_function=exponential)
fit.add_error(axis='x', err_val=xerr)  # add the x-error to the fit
fit.add_error(axis='y', err_val=yerr)  # add the y-errors to the fit

fit.do_fit()  # perform the fit
fit.report(asymmetric_parameter_errors=True)  # print a report with asymmetric uncertainties

# Optional: create a plot
plot = Plot(fit)
plot.plot(asymmetric_parameter_errors=True, ratio=True)  # add the ratio data/function and asymmetric errors

# Optional: create the contours profiler
cpf = ContoursProfiler(fit)
cpf.plot_profiles_contours_matrix()  # plot the contour profile matrix for all parameters

plt.show()
