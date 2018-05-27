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

There are several ways to achieve this with *kafe*. The method chosen
here is to use the :py:object:`~kafe.multifit.Multifit` functionality
to fit both models simultaneously to the :math:`T(U)` and :math:`I(U)`
datasets.

In general, this approach yields different results than the one using
parameter constraints, which is demonstrated in example 11.
"""


import kafe
import numpy as np
import matplotlib.pyplot as plt

# need to import some tools from kafe explicitly
from kafe import ASCII, LaTeX, FitFunction
from kafe.numeric_tools import cov_to_cor  # needed to obtain parameter correlations

# -- Start by defining our models

# empirical model for T(U): a parabola
@ASCII(expression='p2*U^2 + p1*U + p0')
@LaTeX(name='T',
       parameter_names=('p_2', 'p_1', 'p_0'),
       expression=r'p_2 U^2 + p_1 U + p_0')
@FitFunction
def empirical_T_U_model(U, p2=1.0, p1=1.0, p0=0.0):
    # use quadratic model as empirical temerature dependence T(U)
    return p2 * U**2 + p1 * U + p0



# model of current-voltage dependence I(U) for a heating resistor
@ASCII(expression='U/(R0 * (1 + t(p2, p1, p0) * alpha_T)')
@LaTeX(name='I',
       parameter_names=(r'R_0', r'\alpha_T','p_2','p_1','p_0'),
       expression=r'U / \left( R_0 (1 + t(p_i) \cdot \alpha_T) \right)')
@FitFunction
def I_U_model(U, R0=1., alph=0.004, p2=1.0, p1=1.0, p0=0.0):
    # use quadratic model as empirical temerature dependence T(U)
    _temperature = empirical_T_U_model(U, p2, p1, p0)
    # plug the temperature into the model
    return U / (R0 * (1.0 + _temperature * alph))


# -- Next, read the data from an external file

# load all data into numpy arrays
U, I, T = np.loadtxt('OhmsLawExperiment.dat', unpack=True)  # data
sigU, sigI, sigT = 0.1, 0.1, 0.1  # uncertainties

T0 = 273.15  # 0 degrees C as absolute Temperature (in Kelvin)


# -- Finally, go through the fitting procedure

# Step 1:  construct a kafe dataset for the T-U data
kData_T_U = kafe.Dataset(
              data=(U, T-T0),
              basename='u-t-data',
              title='Temperature vs. Voltage',
              axis_labels=['U [V]', r"T [$^\circ$C]"])

# declare errors on U and T
kData_T_U.add_error_source('x', 'simple', sigU)
kData_T_U.add_error_source('y', 'simple', sigT)


# Step 2:  construct a kafe dataset for the U-I data
kData_I_U = kafe.Dataset(
              data=(U, I),
              basename='u-i-data',
              title='Current vs. Voltage',
              axis_labels=['U [V]', 'I [A]'])

# declare errors on U and I
kData_I_U.add_error_source('x', 'simple', sigU)
kData_I_U.add_error_source('y', 'simple', sigI)



# Step 3:  construct and do the multi-model fit
kMultifit_I_T_U_empirical = kafe.Multifit([
                               (kData_T_U, empirical_T_U_model),
                               (kData_I_U, I_U_model)],   
                               fit_name='i-t-u-multifit-empirical')

# identify and link parameters and data which are the same
kMultifit_I_T_U_empirical.autolink_datasets()
kMultifit_I_T_U_empirical.autolink_parameters()

# do the fit
kMultifit_I_T_U_empirical.do_fit()

# plot the results
kPlot_I_U_empirical = kafe.Multiplot(kMultifit_I_T_U_empirical)
kPlot_I_U_empirical.plot_all()


# save and show the resulting fit
kPlot_I_U_empirical.save_all('kafe_example_TU.png', 'kafe_example_IU.png')
plt.show()
