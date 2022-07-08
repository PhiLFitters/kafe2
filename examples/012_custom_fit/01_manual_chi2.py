#!/usr/bin/env python
r"""
Fit a user-defined cost function
================================

This example illustrates an application of *CustomFit*, which does not 
explicitly use data *d* or a model *m*. Instead the user must manually 
define how the cost function value is calculated from the fit 
parameters *p*.

Because any potential data is outside *kafe2* there is no built-in 
visualization (plotting) available except for the fit parameter 
profiles/contours calculated by ContoursProfiler.
"""

import numpy as np
from kafe2 import CustomFit, ContoursProfiler

x_data = np.array([1.0, 2.0, 3.0, 4.0])
x_error = 0.1
y_data = np.array([2.3, 4.2, 7.5, 9.4])
y_error = 0.4


def model_function(x, a, b):
    return a * x + b


def model_function_derivative(x, a, b):
    return a


def chi2(a, b):
    y_model = model_function(x_data, a, b)
    residuals = y_model - y_data

    derivative = model_function_derivative(x_data, a, b)
    xy_error_squared = y_error ** 2 + (derivative * x_error) ** 2

    return np.sum(residuals ** 2 / xy_error_squared) + np.log(np.sum(xy_error_squared))


fit = CustomFit(chi2)
fit.do_fit()
fit.report(asymmetric_parameter_errors=True)

cpf = ContoursProfiler(fit)
cpf.plot_profiles_contours_matrix()

cpf.save()

cpf.show()  # Just a convenience wrapper for matplotlib.pyplot.show()
