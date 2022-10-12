#!/usr/bin/env python
"""
kafe2 example: constraints
==========================

The models used to describe physical phenomena usually depend on a multitude of parameters.
However, for many experiments only one of the parameters is of actual interest to the experimenter.
Still, because model parameters are generally not uncorrelated the experimenter has to factor in
the nuisance parameters for their estimation of the parameter of interest.
Historically this has been done by propagating the uncertainties of the nuisance parameters onto
the y-axis of the data and then performing a fit with those uncertainties.
Thanks to computers, however, this process can also be done numerically by applying parameter
constraints.
This example demonstrates the usage of those constraints in the kafe2 framework.
It also demonstrates how parameters can be limited to specified intervals.

In terms of physics, this example simulates the following experiment:
A steel ball of radius r has been connected to the ceiling by a string of length l, forming a
pendulum.
Due to earth's gravity providing a restoring force this system is a harmonic oscillator.
Because of friction between the steel ball and the surrounding air the oscillator is also damped by
the viscous damping coefficient c.
The goal of the experiment is to determine the local strength of earth's gravity g.
Since the earth is shaped like an ellipsoid the gravitational pull varies with latitude:
it's strongest at the poles with g_p=9.832 m/s^2 and it's weakest at the equator with g_e=9.780
m/s^2.
For reference, at Germany's latitude g lies at approximately 9.81 m/s^2.
"""

import numpy as np

from kafe2 import XYContainer, Fit, Plot

# Relevant physical magnitudes and their uncertainties:
l, delta_l = 10.0, 0.001  # length of the string, l = 10.0+-0.001 m
r, delta_r = 0.052, 0.001  # radius of the steel ball, r = 0.052+-0.001 m
# Amplitude of the steel ball at t=0 in degrees, y_0 = 0.6+-0.006% degrees:
y_0, delta_y_0 = 0.6, 0.01  # Note that the uncertainty on y_0 is relative to y_0
g_0 = 9.81  # Initial guess for g


# Model function for a pendulum as a 1d, damped harmonic oscillator with zero initial speed:
# t = time, y_0 = initial_amplitude, l = length of the string,
# r = radius of the steel ball, g = gravitational acceleration, c = damping coefficient.
def damped_harmonic_oscillator(t, y_0, l, r, g, c):
    # Effective length of the pendulum = length of the string + radius of the steel ball:
    l_total = l + r
    omega_0 = np.sqrt(g / l_total)  # Phase speed of an undamped pendulum.
    omega_d = np.sqrt(omega_0 ** 2 - c ** 2)  # Phase speed of a damped pendulum.
    return y_0 * np.exp(-c * t) * (np.cos(omega_d * t) + c / omega_d * np.sin(omega_d * t))


# Load data from yaml, contains data and errors:
data = XYContainer.from_file(filename='data.yml')

# Create fit object from data and model function:
fit = Fit(data=data, model_function=damped_harmonic_oscillator)

# Constrain model parameters to measurements:
fit.add_parameter_constraint(name='l',   value=l,   uncertainty=delta_l)
fit.add_parameter_constraint(name='r',   value=r,   uncertainty=delta_r)
fit.add_parameter_constraint(name='y_0', value=y_0, uncertainty=delta_y_0, relative=True)

# Lengths between two points are by definition positive, this can be expressed with one-sided limit.
# Note: for technical reasons these limits are inclusive.
fit.limit_parameter("y_0", lower=1e-6)
fit.limit_parameter("l", lower=1e-6)
fit.limit_parameter("r", lower=1e-6)

# Set limits for g that are much greater than the expected deviation but still close to 9.81:
fit.limit_parameter("g", lower=9.71, upper=9.91)

# Solutions are real if c < g / (l + r). Set the upper limit for c a little lower:
c_max = 0.9 * g_0 / (l + r)
fit.limit_parameter("c", lower=1e-6, upper=c_max)

# Optional: Set the initial values of parameters to our initial guesses.
# This can help with convergence, especially when no constraints or limits are specified.
fit.set_parameter_values(y_0=y_0, l=l, r=r, g=g_0)
# Note: this could also be achieved by changing the positional arguments of our model function
# into keyword arguments with our initial guesses as the default values.

# Perform the fit:
fit.do_fit()

# Optional: Print out a report on the fit results on the console.
fit.report(asymmetric_parameter_errors=True)

# Optional: plot the fit results.
plot = Plot(fit)
plot.plot(residual=True, asymmetric_parameter_errors=True)

plot.show()
