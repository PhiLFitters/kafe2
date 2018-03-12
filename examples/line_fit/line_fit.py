#!/usr/bin/env python2
"""
kafe2 example: Fit a line
=========================
"""

import numpy as np

from kafe.fit import XYFit, XYPlot

# import matplotlib *after* kafe2

import matplotlib.pyplot as plt

###################
# Model functions #
###################
def linear_model(x, a, b):
    # our first model is a simple linear function
    return a * x + b

# create XYFits, specifying the measurement data and model function
line_fit = XYFit(xy_data=[[1., 2., 4.], [2.3, 4.2, 9.4]], model_function=linear_model)

# assign LaTeX strings to various quantities (for nice display)
line_fit.assign_parameter_latex_names(a='a', b='b')
line_fit.assign_model_function_latex_expression("{a}{x} + {b}")

line_fit.report()

# perform the fit
line_fit.do_fit()

# print out a report on the result of each fit
line_fit.report()

# to see the fit results, plot using XYPlot
p = XYPlot(fit_objects=line_fit)
p.plot()
p.show_fit_info_box(format_as_latex=True)

# show the fit result
plt.show()


