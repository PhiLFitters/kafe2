#!/usr/bin/env python
# -*- coding: utf8 -*-
r"""
Comparing shared errors with separate errors
--------------------------------------------

When creating a :py:object:`~kafe.fit.multi.Multifit` object for our problem there are two ways to
add errors: Errors can either be added to both fits individually or to both fits at once. Note that
these two procedures are **not** equivalent.

When you add an error to two fits individually you are adding two **different** errors. The error
of the first fit is uncorrelated with the error of the second fit.

When you add an error to both fits at once you are adding the **same** error to both fits. The error
of the first fit will be 100% correlated with the error of the second fit.

In our case we have to add x errors to both fits at once because we have measured I and T at the
same time. The error on U (the x error) affected I and T in exactly the same way. If we had measured
I and T independently of one another the error on U would not be (100%) correlated. In this case we
would have to add a different error to each fit.

To illustrate the importance of the above distinction this example will create two MultiFits: one
with a shared x error and one with two separate x errors. Because the error on U will be treated as
entirely uncorrelated the total x error will be overestimated by a factor of sqrt(2). Also, due to
nonlinear behavior introduced by x errors this will give us an incorrect estimate of the resistance
R0.
"""

########################################################
# This example starts the same as the MultiFit example #
########################################################

import numpy as np

from kafe2 import XYFit, MultiFit


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

#####################################################################
# This is where this example starts to differ from the previous one #
#####################################################################


# We define a method for constructing a MultiFit with either shared or separate x errors:
def construct_multi_fit(shared_x_error):
    fit_1 = XYFit(xy_data=[U, T], model_function=empirical_T_U_model)
    fit_1.add_error(axis='y', err_val=sigT)  # declare errors on T

    fit_2 = XYFit(xy_data=[U, I], model_function=I_U_model)
    fit_2.add_error(axis='y', err_val=sigI)  # declare errors on I

    multi_fit = MultiFit(fit_list=[fit_1, fit_2], minimizer='iminuit')

    if shared_x_error:
        multi_fit.add_error(axis='x', err_val=sigU, fits='all')
    else:
        fit_1.add_error(axis='x', err_val=sigU)
        fit_2.add_error(axis='x', err_val=sigU)

    return multi_fit


multi_fit_separate = construct_multi_fit(shared_x_error=False)
multi_fit_shared = construct_multi_fit(shared_x_error=True)

multi_fit_separate.do_fit()
multi_fit_shared.do_fit()

index_r0 = 3
r0_separate = multi_fit_separate.parameter_values[index_r0]
error_r0_separate = multi_fit_separate.parameter_errors[index_r0]
r0_shared = multi_fit_shared.parameter_values[index_r0]
error_r0_shared = multi_fit_shared.parameter_errors[index_r0]

print("Results for R0:")
print("For separate x errors: %.3f +- %.3f" % (r0_separate, error_r0_separate))
print("For shared x errors: %.3f +- %.3f" % (r0_shared, error_r0_shared))
print("Relative error on R0 from using separate x errors: %.2f%%" %
      (100 * np.abs(r0_separate - r0_shared) / r0_shared))
