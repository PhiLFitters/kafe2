#!/usr/bin/env python
# -*- coding: utf8 -*-
r"""
Fitting several related models in a multi-model fit
===================================================

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
resistane with the temperature into account:

.. math::

    I(U) = \frac{U}{R_0 (1 + t \cdot \alpha_T)}

In the above, :math:`t` is the temperature in degrees Celsius,
:math:`\alpha_T` is an empirical "heat coefficient", and :math:`R_0`
is the resistance at 0 degrees Celsius, which we want to determine.

In essence, there are two models here which must be fitted to the
:math:`I(U)` and :math:`T(U)` data sets, and one model "incorporates"
the other in some way.


Approach 2: multi-model fit
---------------------------

There are several ways to achieve this with *kafe2*. The method chosen
here is to use the :py:object:`~kafe.fit.xy_multi.XYMultifit` functionality
to fit both models simultaneously to the :math:`T(U)` and :math:`I(U)`
datasets.

In general, this approach yields different results than the one using
parameter constraints, which is demonstrated in the example called constraints.
"""


import numpy as np
import matplotlib.pyplot as plt

from kafe2 import XYMultiFit, XYMultiPlot

# empirical model for T(U): a parabola
# independent variable MUST be named x!
def empirical_T_U_model(x, p2=1.0, p1=1.0, p0=0.0):
    # use quadratic model as empirical temerature dependence T(U)
    return p2 * x**2 + p1 * x + p0



# model of current-voltage dependence I(U) for a heating resistor
# independent variable MUST be named x!
def I_U_model(x, R0=1., alph=0.004, p2=1.0, p1=1.0, p0=0.0):
    # use quadratic model as empirical temerature dependence T(U)
    _temperature = empirical_T_U_model(x, p2, p1, p0)
    # plug the temperature into the model
    return x / (R0 * (1.0 + _temperature * alph))


# -- Next, read the data from an external file

# load all data into numpy arrays
U, I, T = np.loadtxt('OhmsLawExperiment.dat', unpack=True)  # data
sigU, sigI, sigT = 0.1, 0.1, 0.1  # uncertainties

T0 = 273.15  # 0 degrees C as absolute Temperature (in Kelvin)
T -= T0 #Measurements are in Kelvin, convert to °C

# -- Finally, go through the fitting procedure

# Step 1:  construct an XYMultiFit object
fit = XYMultiFit(xy_data=[[U, T], [U, I]],
                 model_function=[empirical_T_U_model, I_U_model])

# declare errors on U
fit.add_simple_shared_x_error(err_val=sigU, correlation=0, relative=False)

# declare errors on T
fit.add_simple_error(axis='y', err_val=sigT, model_index=0)

# declare errors on I
fit.add_simple_error(axis='y', err_val=sigI, model_index=1)


# Step 3 (optional): Assign names for models and parameters
fit.assign_parameter_latex_names(x='U', p2='p_2', p1='p_1', p0='p_0', R0='R_0', alph=r'\alpha_\mathrm{T}')

fit.assign_model_function_expression('{0}*{x}^2 + {1}*{x} + {2}', model_index=0)
fit.assign_model_function_latex_expression(r'{0}\,{x}^2 + {1}\,{x} + {2}', model_index=0)
fit.assign_model_function_expression('{x} / ({0} * (1 + ({2}*{x}^2 + {3}*{x} + {4}) * {1}))', model_index=1)
fit.assign_model_function_latex_expression(r'\frac{{{x}}}{{{0} \cdot (1 + ({2}{x}^2 + {3}{x} + {4}) \cdot {1})}}', model_index=1)

# Step 4: do the fit
fit.do_fit()

#(Optional) print the results
fit.report()

#(Optional) plot the results
plot = XYMultiPlot(fit, separate_plots=True)
plot.plot()
plot.show_fit_info_box(format_as_latex=True)
plt.show()
