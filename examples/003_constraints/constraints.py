#!/usr/bin/env python2
"""
kafe2 example: constraints
==========================

The models used to describe physical phenomena usually depend on a multitude of parameters.
However, for many experiments only one of the parameters is of actual interest to the experimenter.
Still, because model parameters are generally not uncorrelated the experimenter has to factor in the nuisance parameters
for their estimation of the parameter of interest.
Historically this has been done by propagating the uncertainties of the nuisance parameters onto the y-axis of the data
and then performing a fit with those uncertainties.
Thanks to computers, however, this process can also be done numerically by applying parameter constraints.
This example demonstrates the usage of those constraints in the kafe2 framework.

More specifically, this example will simulate the following experiment:
A steel ball of radius r has been connected to the ceiling by a string of length l, forming a pendulum.
Due to earth's gravity providing a restoring force this system is a harmonic oscillator.
Because of friction between the steel ball and the surrounding air the oscillator is also damped by the viscous damping
coefficient c.
The goal of the experiment is to determine the local strength of earth's gravity g.
Since the earth is shaped like an ellipsoid the gravitational pull varies with latitude: it's strongest at the poles
with g_p=9,780 m/s^2 and it's weakest at the equator with g_e=9.832 m/s^2.
For reference, at Germany's latitude g lies at approximately 9.81 m/s^2.
"""

import numpy as np
import matplotlib.pyplot as plt

from kafe2 import XYContainer, XYFit, XYPlot

# Relevant physical magnitudes and their uncertainties
l, delta_l = 10.0, 0.001  # length of the string, l = 10.0+-0.001 m
r, delta_r = 0.052, 0.001  # radius of the steel ball, r = 0.052+-0.001 kg
# Note that the uncertainty on y_0 is relative to y_0
y_0, delta_y_0 = 0.6, 0.01  # amplitude of the steel ball at x=0 in degrees, y_0 = 0.6+-0.006% degrees


# Model function for a pendulum as a one-dimensional, damped harmonic oscillator with zero initial speed
# x = time, y_0 = initial_amplitude, l = length of the string,
# r = radius of the steel ball, g = gravitational acceleration, c = damping coefficient
def damped_harmonic_oscillator(x, y_0, l, r, g, c):
    l_total = l + r  # effective length of the pendulum = length of the string + radius of the steel ball
    omega_0 = np.sqrt(g / l_total)  # phase speed of an undamped pendulum
    omega_d = np.sqrt(omega_0 ** 2 - c ** 2)  # phase speed of a damped pendulum
    return y_0 * np.exp(-c * x) * (np.cos(omega_d * x) + c / omega_d * np.sin(omega_d * x))


# Load data from yaml, contains data and errors
data = XYContainer.from_file(filename='data.yml')

# Create fit object from data and model function
fit = XYFit(xy_data=data, model_function=damped_harmonic_oscillator)

# Constrain model parameters to measurements
fit.add_parameter_constraint(name='l',   value=l,   uncertainty=delta_l)
fit.add_parameter_constraint(name='r',   value=r,   uncertainty=delta_r)
fit.add_parameter_constraint(name='y_0', value=y_0, uncertainty=delta_y_0, relative=True)

# Because the model function is oscillating the fit needs to be initialized with near guesses for unconstrained
# parameters in order to converge
g_initial = 9.81  # initial guess for g
c_initial = 0.01  # initial guess for c
fit.set_parameter_values(g=g_initial, c=c_initial)

# Optional: Print out a report on the fit results on the console.
fit.report()

# Perform the fit
fit.do_fit()

# Optional: plot the fit results
plot = XYPlot(fit)
plot.plot()
plot.show_fit_info_box()

plt.show()