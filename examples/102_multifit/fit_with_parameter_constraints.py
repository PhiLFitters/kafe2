#!/usr/bin/env python
# -*- coding: utf8 -*-
r"""
Fitting several related models using parameter constraints
==========================================================

The premise of this example is deceptively simple: a series
of voltages is applied to a resistor and the resulting current
is measured. The aim is to fit a model to the collected data
consisting of voltage-current pairs and determine the
resistance :math:`R`.

According to Ohm's Law, the relation between current and voltage
is linear, so a linear model can be fitted. However, Ohm's Law
only applies to an ideal resistor whose resistance does not
change, and the resistance of a real resistor tends to increase
as the resistor heats up. This means that, as the applied voltage
gets higher, the resistance changes, giving rise to
nonlinearities which are ignored by a linear model.

To get a hold on this nonlinear behavior, the model must take
the temperature of the resistor into account. Thus, the
temperature is also recorded for every data point.
The data thus consists of triples, instead of the usual "xy" pairs,
and the relationship between temperature and voltage must be
modeled in addition to the one between current and voltage.

Here, the dependence :math:`T(U)` is taken to be quadratic, with
some coefficients :math:`p_0`, :math:`p_1`, and :math:`p_2`:

.. math::

    T(U) = p_2 U^2 + p_1 U + p_0

This model is based purely on empirical observations. The :math:`I(U)`
dependence is more complicated, but takes the "running" of the
resistance with the temperature into account:

.. math::

    I(U) = \frac{U}{R_0 (1 + t \cdot \alpha_T)}

In the above, :math:`t` is the temperature in degrees Celsius,
:math:`\alpha_T` is an empirical "heat coefficient", and :math:`R_0`
is the resistance at 0 degrees Celsius, which we want to determine.

In essence, there are two models here which must be fitted to the
:math:`I(U)` and :math:`T(U)` data sets, and one model "incorporates"
the other in some way.


Approach 1: parameter constraints
---------------------------------

There are several ways to achieve this with *kafe2*. The method chosen
here consists of two steps: First, a quadratic model is fitted to the
:math:`T(U)` datasets to estimate the parameters :math:`p_0`, :math:`p_1`
and :math:`p_2` and their covariance matrix.

Next, the :math:`I(U)` model is fitted, with the temperature :math:`t`
being explicitly replaced by its parameterization as a function of
:math:`p_0`, :math:`p_1` and :math:`p_2`. The key here is to fit these
parameters again from the :math:`I(U)` dataset, but to constrain them
to the values obtained in the previous :math:`T(U)` fit.

In general, this approach yields different results than the one using
a simultaneous multi-model fit, which is demonstrated in the example called
``multifit``.
"""


import numpy as np
import matplotlib.pyplot as plt

from kafe2 import XYFit, Plot


# empirical model for T(U): a parabola
# independent variable MUST be named x!
def empirical_T_U_model(x, p2=1.0, p1=1.0, p0=0.0):
    # use quadratic model as empirical temperature dependence T(U)
    return p2 * x**2 + p1 * x + p0


# model of current-voltage dependence I(U) for a heating resistor
# independent variable MUST be named x!
def I_U_model(x, R0=1., alph=0.004, p2=1.0, p1=1.0, p0=0.0):
    # use quadratic model as empirical temperature dependence T(U)
    _temperature = empirical_T_U_model(x, p2, p1, p0)
    # plug the temperature into the model
    return x / (R0 * (1.0 + _temperature * alph))


# -- Next, read the data from an external file

# load all data into numpy arrays
U, I, T = np.loadtxt('OhmsLawExperiment.dat', unpack=True)  # data
sigU, sigI, sigT = 0.1, 0.1, 0.1  # uncertainties

T0 = 273.15  # 0 degrees C as absolute Temperature (in Kelvin)
T -= T0  # Measurements are in Kelvin, convert to °C

# -- Finally, go through the fitting procedure

# Step 1: perform an "auxiliary" fit to the T(U) data
auxiliary_fit = XYFit(
    xy_data=[U, T],
    model_function=empirical_T_U_model
)

# (Optional): Assign names for models and parameters
auxiliary_fit.assign_parameter_latex_names(x='U', p2='p_2', p1='p_1', p0='p_0')
auxiliary_fit.assign_model_function_expression('{0}*{x}^2 + {1}*{x} + {2}')
auxiliary_fit.assign_model_function_latex_expression(r'{0}\,{x}^2 + {1}\,{x} + {2}')

# declare errors on U
auxiliary_fit.add_simple_error(axis='x', err_val=sigU)

# declare errors on T
auxiliary_fit.add_simple_error(axis='y', err_val=sigT)

# perform the auxiliary fit
auxiliary_fit.do_fit()

# (Optional) print the results
auxiliary_fit.report()

# (Optional) plot the results
auxiliary_plot = Plot(auxiliary_fit)
auxiliary_plot.plot(with_fit_info=True)

# Step 2: perform the main fit
main_fit = XYFit(
    xy_data=[U, I],
    model_function=I_U_model
)

# declare errors on U
main_fit.add_simple_error(axis='x', err_val=sigU)
# declare errors on I
main_fit.add_simple_error(axis='y', err_val=sigI)

# constrain the parameters
main_fit.add_matrix_parameter_constraint(
    names=auxiliary_fit.parameter_names,
    values=auxiliary_fit.parameter_values,
    matrix=auxiliary_fit.parameter_cov_mat,
    matrix_type='cov'  # default matrix type is cov, this kwarg is just for clarity
)

# (Optional): Assign names for models and parameters
main_fit.assign_parameter_latex_names(x='U', p2='p_2', p1='p_1', p0='p_0', R0='R_0', alph=r'\alpha_\mathrm{T}')
main_fit.assign_model_function_expression('{x} / ({0} * (1 + ({2}*{x}^2 + {3}*{x} + {4}) * {1}))')
main_fit.assign_model_function_latex_expression(r'\frac{{{x}}}{{{0} \cdot (1 + ({2}{x}^2 + {3}{x} + {4}) \cdot {1})}}')

# Step 4: do the fit
main_fit.do_fit()

# (Optional) print the results
main_fit.report()

# (Optional) plot the results
plot = Plot(main_fit)
plot.plot(with_fit_info=True)

plt.show()
