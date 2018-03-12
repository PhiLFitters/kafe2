#!/usr/bin/env python2
"""
kafe2 example: Fit a double slit diffraction pattern
====================================================
"""

import numpy as np

from kafe.fit import XYContainer, XYFit, XYPlot

# import matplotlib *after* kafe2
import matplotlib.pyplot as plt

def _generate_dataset(output_filename='data.yml'):
    """
    Create an XYContainer holding the measurement data
    and the errors and write it to a file.
    """

    xy_data = [
        [ # x data
            9.574262e-01,
            2.262212e+00,
            3.061167e+00,
            3.607280e+00,
            4.933100e+00,
            5.992332e+00,
            7.021234e+00,
            8.272489e+00,
            9.250817e+00,
            9.757758e+00,
        ],
        [ # y data
            1.672481e+00,
            1.743410e+00,
            1.805217e+00,
            2.147802e+00,
            2.679615e+00,
            3.110055e+00,
            3.723173e+00,
            4.430122e+00,
            4.944116e+00,
            5.698063e+00,
        ]
    ]

    d = XYContainer(x_data=xy_data[0],
                    y_data=xy_data[1])

    d.add_simple_error('x', 0.3, relative=False)
    d.add_simple_error('y', 0.2, relative=False)

    d.to_file(output_filename)

###################
# Model functions #
###################
def linear_model(x, a, b):
    # our first model is a simple linear function
    return a * x + b

def exponential_model(x, A0=1., x0=5.):
    # our second model is a simple exponential function
    return A0 * np.exp(x/x0)


# read in the measurement data from a file
d = XYContainer.from_file("two_models.yml")

# create XYFits, specifying the measurement data and model function
linear_fit = XYFit(xy_data=d, model_function=linear_model)
exponential_fit = XYFit(xy_data=d, model_function=exponential_model)

# assign LaTeX strings to various quantities (for nice display)
linear_fit.assign_parameter_latex_names(a='a', b='b')
linear_fit.assign_model_function_latex_expression("{a}{x} + {b}")
exponential_fit.assign_parameter_latex_names(A0='A_0', x0='x_0')
exponential_fit.assign_model_function_latex_expression("{A0} e^{{{x}/{x0}}}")

# perform the fits
linear_fit.do_fit()
exponential_fit.do_fit()

# print out a report on the result of each fit
linear_fit.report()
exponential_fit.report()

# to see the fit results, plot using XYPlot
p = XYPlot(fit_objects=[linear_fit, exponential_fit])
p.plot()
p.show_fit_info_box(format_as_latex=True)

# show the fit result
plt.show()


