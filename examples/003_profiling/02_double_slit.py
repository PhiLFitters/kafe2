#!/usr/bin/env python
"""
kafe2 Example: Fit of a Double Slit Diffraction Pattern
=======================================================

When the monochromatic light emitted by a laser hits a double slit the interference of the two slits
results in a characteristic refraction pattern. This refraction pattern is based on trigonometric
functions and is therefore highly non-linear in its parameters (see the previous example for an
explanation of the implications). In this example the Fraunhofer diffraction equation is fitted to
the intensity of laser light
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2.fit import XYContainer, Fit, Plot
from kafe2.fit.tools import ContoursProfiler


def _generate_dataset(output_filename='02_double_slit_data.yml'):
    """
    Create an XYContainer holding the measurement data
    and the errors and write it to a file.
    """

    xy_data = [[  # x data: position
                -0.044, -0.040, -0.036, -0.030, -0.024, -0.018, -0.012, -0.008, -0.004, -0.001,
                 0.004,  0.008,  0.012,  0.018,  0.024,  0.030,  0.036,  0.040,  0.044],
               [  # y data: light intensity
                0.06, 0.07, 0.03, 0.04, 0.32, 0.03, 0.64, 0.08, 0.20, 1.11, 0.52, 0.07, 0.89, 0.01,
                0.17, 0.05, 0.09, 0.02, 0.01]]

    data = XYContainer(x_data=xy_data[0], y_data=xy_data[1])

    data.add_error('x', 0.002, relative=False, name='x_uncor_err')
    data.add_error(
        'y',
        [0.02, 0.02, 0.02, 0.02, 0.04, 0.02, 0.05, 0.03, 0.05, 0.08, 0.05, 0.03, 0.05, 0.01, 0.04,
         0.03, 0.03, 0.02, 0.01],
        relative=False,
        name='y_uncor_err'
    )

    data.to_file(output_filename)

# Uncomment this to re-generate the dataset:
# _generate_dataset()


def intensity(theta, i_0, b, g, wavelength):
    """
    In this example our model function is the intensity of diffracted light as described by the
    Fraunhofer equation.
    :param theta: angle at which intensity is measured
    :param i_0: intensity amplitude
    :param b: width of a single slit
    :param g: distance between the two slits
    :param wavelength: wavelength of the laser light
    :return: intensity of the diffracted light
    """
    single_slit_arg = np.pi * b * np.sin(theta) / wavelength
    single_slit_interference = np.sin(single_slit_arg) / single_slit_arg
    double_slit_interference = np.cos(np.pi * g * np.sin(theta) / wavelength)
    return i_0 * single_slit_interference ** 2 * double_slit_interference ** 2


# Read in the measurement data from the file generated above:
data = XYContainer.from_file("02_double_slit_data.yml")

# Create fit from data container:
fit = Fit(data=data, model_function=intensity, minimizer="iminuit")

# Optional: assign LaTeX names for prettier fit info box:
fit.assign_parameter_latex_names(theta=r'\theta', i_0='I_0', b='b', g='g', wavelength=r'\lambda')
fit.assign_model_function_latex_name('I')
fit.assign_model_function_latex_expression(
    r"{i_0}\,\left(\frac{{\sin(\frac{{\pi}}{{{wavelength}}}\,b\,\sin{{{theta}}})}}"
    r"{{\frac{{\pi}}{{{wavelength}}}\,b\,\sin{{{theta}}}}}"
    r"\cos(\frac{{\pi}}{{{wavelength}}}\,g\,\sin{{{theta}}})\right)^2"
)

# Limit parameters to positive values to better model physical reality:
eps = 1e-8
fit.limit_parameter('i_0', lower=eps)
fit.limit_parameter('b', lower=eps)
fit.limit_parameter('g', lower=eps)

# Set fit parameters to near guesses to improve convergence:
fit.set_parameter_values(i_0=1., b=20e-6, g=50e-6)

# The fit parameters have no preference in terms of values.
# Their profiles are highly distorted, indicating a very non-linear fit.
# You can try constraining them via external measurements to make the fit more linear:
# f.add_parameter_constraint('b', value=13.5e-6, uncertainty=1e-6)
# f.add_parameter_constraint('g', value=50e-6, uncertainty=1e-6)

# Fix the laser wavelength to 647.1 nm (krypton laser) since its uncertainty is negligible:
fit.fix_parameter('wavelength', value=647.1e-9)

# Fit objects can also be saved to files:
fit.to_file('02_double_slit.yml')
# The generated file can be used as input for kafe2go.

# Alternatively you could load it back into code via:
# f = XYFit.from_file('02_double_slit.yml')

fit.do_fit()

cpf = ContoursProfiler(fit)
cpf.plot_profiles_contours_matrix(parameters=['i_0', 'b', 'g'],
                                  show_grid_for='all',
                                  show_fit_minimum_for='all',
                                  show_error_span_profiles=True,
                                  show_legend=True,
                                  show_parabolic_profiles=True,
                                  show_ticks_for='all',
                                  contour_naming_convention='sigma',
                                  label_ticks_in_sigma=True)

# To see the fit results, plot using Plot:
p = Plot(fit_objects=fit)
p.x_label = r"$\theta$"
p.y_label = r"$I$"
p.plot(asymmetric_parameter_errors=True)

# Show the fit results:
plt.show()
