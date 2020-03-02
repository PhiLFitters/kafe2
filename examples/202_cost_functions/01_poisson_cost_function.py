#!/usr/bin/env python
"""
kafe2 example: Poisson cost function
====================================

In data analysis the uncertainty on measurement data is most often assumed to resemble a normal distribution.
For many use cases this assumption works reasonably well but there is a problem: to get meaningful fit results
you need to know about the uncertainties of your measurements. Now imagine for a moment that the quantity you're
measuring is the number of radioactive decays coming from some substance in a given time period. What is your
data error in this case? The precision with that you can correctly count the decays? The answer is that due to
the inherently random nature of radioactive decay the variance, and therefore the uncertainty on your measurement
data directly follows from the mean number of decays in a given time period - the number of decays are following
a poisson distribution. In kafe2 this distribution can be modeled by initializing a fit object with a special
cost function. In previous examples when no cost function was provided a normal distribution has been assumed
by default. It is important to know that for large numbers of events a poisson distribution can be approximated
by a normal distribution (y_error = sqrt(y_data)). Consult the other examples in this folder for more details.

For our example on cost functions we imagine the following, admittedly a little contrived scenario:
In some remote location on earth archeologists have found the ruins of an ancient civilization. They estimate
the ruins to be about 7000 years old. The civilization in question seems to have known about mathematics and they
even had their own calendar. Unfortunately we do not know the exact offset of this ancient calendar relative to
our modern calendar. Luckily the ancient civilization seems to have mummified their rulers and written down their
years of death though. Using a method called radiocarbon dating we can now try to estimate the offset between the
ancient and the modern calendar by analyzing the relative amounts of carbon isotopes in the mummified remains of
the ancient kings. More specifically, we take small samples from the mummies, extract the carbon from those samples
and then measure the number of decaying carbon-14 atoms in our samples. Carbon-14 is a trace radioisotope with a
half life of only 5730 years that is continuously being produced in earth's upper atmosphere. In a living organism
there is a continuous exchange of carbon atoms with its environment which results in a stable concentration of
carbon-14. Once an organism dies, however, the carbon atoms in its body are fixed and the concentration of
carbon-14 starts to exponentially decrease over time. If we then measure the concentration of carbon-14 in our
samples we can then calculate at which point in time they must have contained atmospheric amounts of carbon-14,
i.e. the times of death of the ancient kings.
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYFit, XYCostFunction_NegLogLikelihood, Plot

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


# This is where we tell the fit to assume a poisson distribution for our data.
xy_fit = XYFit(
    xy_data=[years_of_death, measured_c14_activity],
    model_function=expected_activity_per_day,
    cost_function=XYCostFunction_NegLogLikelihood(data_point_distribution='poisson')
)

# The half life of carbon-14 is only known with a precision of +-40 years
xy_fit.add_parameter_constraint(name='T_12_C14', value=5730, uncertainty=40)

# Perform the fit
# Note that since for a Poisson distribution the data error is directly linked to the mean.
# Because of this fits can be performed without explicitly adding data errors.
xy_fit.do_fit()

# Optional: print out a report on the fit results on the console
xy_fit.report()

# Optional: create a plot of the fit results using Plot
xy_plot = Plot(xy_fit)
xy_plot.plot(fit_info=True)

plt.show()
