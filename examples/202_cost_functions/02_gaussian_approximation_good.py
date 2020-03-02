#!/usr/bin/env python
"""
kafe2 example: Gaussian approximation of Poisson distributions for large N
==========================================================================

This example is a continuation of the Poisson cost function example. While a Poisson distribution with mean N is
inherently asymmetric, it can be approximated using a symmetric Gaussian distribution with a standard deviation of
sqrt(N). This approximation is bad for small N, but for large N it is accurate. In this example we show the Gaussian
approximation working well for large N.
"""

#################################################
# Same as in the previous example, scroll down. #
#################################################

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYFit, Plot

# Years of death are our x-data, measured c14 activity is our y-data.
# Note that our data does NOT include any x or y errors.
years_of_death, measured_c14_activity = np.loadtxt('measured_c14_activity.txt')

days_per_year = 365.25  # assumed number of days per year
current_year = 2019  # current year according to the modern calendar
sample_mass = 1.0  # Mass of the carbon samples in g
initial_c14_concentration = 1e-12  # Assumed initial concentration
N_A = 6.02214076e23  # Avogadro constant in 1/mol
molar_mass_c14 = 14.003241  # Molar mass of the Carbon-14 isotope in g/mol

expected_initial_num_c14_atoms = initial_c14_concentration * N_A * sample_mass / molar_mass_c14


# x = years of death in the ancient calendar
# Delta_t = difference between the ancient and the modern calendar in years
# T_12_C14 = half life of carbon-14 in years, read as T 1/2 carbon-14
def expected_activity_per_day(x, Delta_t=5000, T_12_C14=5730):
    # activity = number of radioactive decays
    expected_initial_activity_per_day = expected_initial_num_c14_atoms * np.log(2) / (T_12_C14 * days_per_year)
    total_years_since_death = Delta_t + current_year - x
    return expected_initial_activity_per_day * np.exp(-np.log(2) * total_years_since_death / T_12_C14)

#########################################
# This is where things start to change. #
#########################################

# We define a fit as per normal, with xy data and a model function.
# Since no cost function is provided it will use the default (chi2 -> Gaussian data errors).
xy_fit = XYFit(
    xy_data=[years_of_death, measured_c14_activity],
    model_function=expected_activity_per_day
)

# We use the Gaussian approximation of the Poisson distribution sqrt(y) for our y data error
xy_fit.add_simple_error(axis='y', err_val=np.sqrt(measured_c14_activity))

# The half life of carbon-14 is only known with a precision of +-40 years
xy_fit.add_parameter_constraint(name='T_12_C14', value=5730, uncertainty=40)

# Perform the fit
xy_fit.do_fit()

# Optional: print out a report on the fit results on the console
xy_fit.report()

# Optional: create a plot of the fit results using Plot
xy_plot = Plot(xy_fit)
xy_plot.plot(fit_info=True)

plt.show()
