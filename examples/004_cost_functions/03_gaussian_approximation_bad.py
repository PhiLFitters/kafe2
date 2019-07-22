#!/usr/bin/env python2
"""
kafe2 example: Gaussian approximation of Poisson distributions for small N
==========================================================================

This example is a continuation of the Poisson cost function example. While a Poisson distribution with mean N is
inherently asymmetric, it can be approximated using a symmetric Gaussian distribution with a standard deviation of
sqrt(N). This approximation is bad for small N, but for large N it is accurate. In this example we show the Gaussian
approximation working poorly for small N. Instead of using the data measured over a 24 hour period we only use the data
measured in the first two minutes of the experiment. While this is sort of an artificial restriction for our imagined
experiment there are experiments in high energy physics where you'll only record a handful of events over the course of
a whole year, if any. Since you can hardly keep the experiment running for a hundred years you will need to make do with
what little data you have - and at that point the difference between a Poisson distribution and its Gaussian
approximation becomes important.
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYFit, XYPlot, XYCostFunction_NegLogLikelihood

# Years of death are our x-data, measured c14 activity is our y-data.
# Note that our data does NOT include any x or y errors.
# Unlike in the previous example we load the sparse data that includes only the first five minutes of the experiment
years_of_death, measured_c14_activity_sparse = np.loadtxt('measured_c14_activity_sparse.txt')

days_per_year = 365.25  # assumed number of two-minute-periods per year
two_min_per_year = days_per_year * 24 * 60 / 2  # assumed number of two-minute-periods per year
current_year = 2019  # current year according to the modern calendar
sample_mass = 1.0  # Mass of the carbon samples in g
initial_c14_concentration = 1e-12  # Assumed initial concentration
N_A = 6.02214076e23  # Avogadro constant in 1/mol
molar_mass_c14 = 14.003241  # Molar mass of the Carbon-14 isotope in g/mol

expected_initial_num_c14_atoms = initial_c14_concentration * N_A * sample_mass / molar_mass_c14


# x = years of death in the ancient calendar
# Delta_t = difference between the ancient and the modern calendar in years
# T_12_C14 = half life of carbon-14 in years, read as T 1/2 carbon-14
def expected_activity_per_two_min(x, Delta_t=5000, T_12_C14=5730):
    # activity = number of radioactive decays
    expected_initial_activity_per_day = expected_initial_num_c14_atoms * np.log(2) / (T_12_C14 * two_min_per_year)
    total_years_since_death = Delta_t + current_year - x
    return expected_initial_activity_per_day * np.exp(-np.log(2) * total_years_since_death / T_12_C14)


# We define a fit as per normal, with xy data and a model function.
# Since no cost function is provided it will use the default (chi2 -> Gaussian data errors).
xy_fit_gaussian_sparse = XYFit(
    xy_data=[years_of_death, measured_c14_activity_sparse],
    model_function=expected_activity_per_two_min
)

# We use the Gaussian approximation of the Poisson distribution sqrt(y) for our y data error
# Because we only have about 10 events per measurement this approximation will be bad.
xy_fit_gaussian_sparse.add_simple_error(axis='y', err_val=np.sqrt(measured_c14_activity_sparse))

# The half life of carbon-14 is only known with a precision of +-40 years
xy_fit_gaussian_sparse.add_parameter_constraint(name='T_12_C14', value=5730, uncertainty=40)

# Perform the fit
xy_fit_gaussian_sparse.do_fit()

# We create another fit, this time with a Poisson distribution, to compare the Gaussian approximation against.
xy_fit_poisson_sparse = XYFit(
    xy_data=[years_of_death, measured_c14_activity_sparse],
    model_function=expected_activity_per_two_min,
    cost_function=XYCostFunction_NegLogLikelihood(data_point_distribution='poisson')
)
xy_fit_poisson_sparse.add_parameter_constraint(name='T_12_C14', value=5730, uncertainty=40)
xy_fit_poisson_sparse.do_fit()

# We calculate and print out the relative error on our estimate of Delta_T from using a Gaussian approximation.
Delta_T_poisson_sparse = xy_fit_poisson_sparse.parameter_values[0]
Delta_T_gaussian_sparse = xy_fit_gaussian_sparse.parameter_values[0]
relative_error_sparse = abs((Delta_T_gaussian_sparse - Delta_T_poisson_sparse) / Delta_T_poisson_sparse)

# For comparison we also calculate the relative error when using the full dataset.
years_of_death, measured_c14_activity_full = np.loadtxt('measured_c14_activity.txt')


def expected_activity_per_day(x, Delta_t=5000, T_12_C14=5730):
    # activity = number of radioactive decays
    expected_initial_activity_per_day = expected_initial_num_c14_atoms * np.log(2) / (T_12_C14 * days_per_year)
    total_years_since_death = Delta_t + current_year - x
    return expected_initial_activity_per_day * np.exp(-np.log(2) * total_years_since_death / T_12_C14)


xy_fit_gaussian_full = XYFit(
    xy_data=[years_of_death, measured_c14_activity_full],
    model_function=expected_activity_per_day
)
xy_fit_gaussian_full.add_simple_error(axis='y', err_val=np.sqrt(measured_c14_activity_full))
xy_fit_gaussian_full.add_parameter_constraint(name='T_12_C14', value=5730, uncertainty=40)
xy_fit_gaussian_full.do_fit()

xy_fit_poisson_full = XYFit(
    xy_data=[years_of_death, measured_c14_activity_full],
    model_function=expected_activity_per_day,
    cost_function=XYCostFunction_NegLogLikelihood(data_point_distribution='poisson')
)
xy_fit_poisson_full.add_parameter_constraint(name='T_12_C14', value=5730, uncertainty=40)
xy_fit_poisson_full.do_fit()

Delta_T_poisson_full = xy_fit_poisson_full.parameter_values[0]
Delta_T_gaussian_full = xy_fit_gaussian_full.parameter_values[0]
relative_error_full = abs((Delta_T_gaussian_full - Delta_T_poisson_full) / Delta_T_poisson_full)

# Print out the results:
print('Relative errors from Gaussian approximation:')
print('For N~10:', relative_error_sparse)
print('For N~6000:', relative_error_full)

# Optional: create a plot of the fit results using XYPlot
xy_plot_poisson_sparse = XYPlot([xy_fit_gaussian_sparse, xy_fit_poisson_sparse])
xy_plot_poisson_sparse.plot()
xy_plot_poisson_sparse.show_fit_info_box()
plt.show()
