#!/usr/bin/env python
"""
kafe2 example: Fit a double slit diffraction pattern
====================================================
"""

import numpy as np

from kafe2.fit import XYContainer, XYFit, Plot
from kafe2.fit.tools import ContoursProfiler

# import matplotlib *after* kafe2
import matplotlib.pyplot as plt

def _generate_dataset(output_filename='double_slit.yml'):
    """
    Create an XYContainer holding the measurement data
    and the errors and write it to a file.
    """

    xy_data = [
        [ # x data: position
            -0.044,
            -0.040,
            -0.036,
            -0.030,
            -0.024,
            -0.018,
            -0.012,
            -0.008,
            -0.004,
            -0.001,
             0.004,
             0.008,
             0.012,
             0.018,
             0.024,
             0.030,
             0.036,
             0.040,
             0.044

        ],
        [ # y data: light intensity
            0.06,
            0.07,
            0.03,
            0.04,
            0.32,
            0.03,
            0.64,
            0.08,
            0.20,
            1.11,
            0.52,
            0.07,
            0.89,
            0.01,
            0.17,
            0.05,
            0.09,
            0.02,
            0.01
        ]
    ]

    d = XYContainer(x_data=xy_data[0],
                    y_data=xy_data[1])

    d.add_error('x', 0.002, relative=False)
    d.add_error('y', [0.02, 0.02, 0.02, 0.02, 0.04, 0.02, 0.05, 0.03, 0.05,
                      0.08, 0.05, 0.03, 0.05, 0.01, 0.04, 0.03, 0.03, 0.02, 0.01], relative=False)

    d.to_file(output_filename)

#_generate_dataset()

###################
# Model functions #
###################
def interference(x, I0=1., b=1e-5, g=2e-5, k=1e7):
    # our first model is a simple linear function
    k_half_sine_alpha = k / 2 * np.sin(x)  # helper variable
    k_b = k_half_sine_alpha * b
    k_g = k_half_sine_alpha * g
    return I0 * (np.sin(k_b) / (k_b) * np.cos(k_g)) ** 2


# read in the measurement data from a file
d = XYContainer.from_file("double_slit.yml")

# create XYFits, specifying the measurement data and model function
f = XYFit(xy_data=d, model_function=interference, minimizer='iminuit')

# assign LaTeX strings to various quantities (for nice display)
f.assign_parameter_latex_names(I0='I_0', b='b', g='g', k='k')
f.assign_model_function_latex_name('I')
f.assign_model_function_latex_expression(
    r"{I0}\,\left(\frac{{\sin(\frac{{{k}}}{{2}}\,b\,\sin{{{x}}})}}"
    r"{{\frac{{{k}}}{{2}}\,b\,\sin{{{x}}}}}"
    r"\cos(\frac{{{k}}}{{2}}\,g\,\sin{{{x}}})\right)^2"
)

# perform the fits
f.set_parameter_values(I0=1., b=20e-6, g=50e-6, k=9.67e6)
f.fix_parameter('k')
f.do_fit()

cpf = ContoursProfiler(f)
cpf.plot_profiles_contours_matrix(parameters=['I0', 'b', 'g'],
                                  show_grid_for='all',
                                  show_fit_minimum_for='all',
                                  show_error_span_profiles=True,
                                  show_legend=True,
                                  show_parabolic_profiles=True,
                                  show_ticks_for='all',
                                  contour_naming_convention='sigma',
                                  label_ticks_in_sigma=True)

# to see the fit results, plot using Plot
p = Plot(fit_objects=f)
p.plot(fit_info=True)

# show the fit result
plt.show()
