"""
kafe2 example: One-sided Limit
==============================

This example aims to illustrate a common problem in particle physics: if there are vew signal events
but a lot of background events then it becomes difficult to separate the signal events from the
statistical fluctuations of the background. If the number of signal events N_s is one of the fit
parameters then N_s can be very close to zero or even negative. In the former case a confidence
level for the non-negative region should be calculated to determine whether there is any signal at
all.
"""

import numpy as np
from scipy.stats import norm
from kafe2 import HistFit, plot, ContoursProfiler

# For this example toy data is generated using NumPy. The parameters below work to demonstrate the
# problem but feel free to try tweaking them.
X_MIN = 0  # Minimal x value for the data.
X_MAX = 20  # Maximal x value for the data.
N_SIGNAL = 50  # Number of signal events.
N_BACKGROUND = 3000  # Number of background events.
# N_SIGNAL = 200
# N_BACKGROUND = 2000
np.random.seed(0)  # Seed for the random number generator.

x_span = X_MAX - X_MIN
x_center = (X_MIN + X_MAX) / 2

# Constant Background:
data_background = np.random.uniform(low=X_MIN, high=X_MAX, size=N_BACKGROUND)
# Gauss-shaped signal peak:
data_signal = np.random.normal(loc=x_center, scale=0.05*x_span, size=N_SIGNAL)
data_combined = np.concatenate((data_background, data_signal))

# NumPy histograms can be used as input data for HistFit:
data_histogram = np.histogram(data_combined, bins=x_span, range=(X_MIN, X_MAX))


# N_b=number of background events, N_s=number of signal events,
# mu_s=mean of the signal peak, sigma_s=standard deviation of the signal peak
def signal_plus_background_model(x, N_b, N_s, mu_s=x_center, sigma_s=0.05*x_span):
    return N_b / (X_MAX - X_MIN) + N_s * norm.pdf(x, mu_s, sigma_s)


# Set density=False because the numbers of events are fit parameters:
fit = HistFit(data_histogram, signal_plus_background_model, density=False)
# fit.limit_parameter("mu_s", lower=(x_center+X_MIN*2)/3, upper=(x_center+X_MAX*2)/3)
# fit.limit_parameter("sigma_s", lower=0, upper=0.1*x_span)

# Constraints are needed to avoid convergence problems for N_s->0:
fit.add_parameter_constraint("mu_s", value=x_center, uncertainty=0.1*x_center)
fit.add_parameter_constraint("sigma_s", value=0.05*x_span, uncertainty=0.005*x_span)
# In real life such constraints may not be available. Parameter limits can also work but with weird
# behavior close to the unphysical region of N_s < 0.

fit.do_fit()

# Create a one-sided profile plot for N_s using ContoursProfiler:
cpf = ContoursProfiler(fit)
cpf.plot_profile(
    "N_s",  # Parameter name
    low=0,  # Lower bound for the profile plot.
    cl=[0.9, 0.95, 0.99]  # Confidence levels for the upper bounds.
)

# Do a regular plot of the fit:
plot(fit, profile=True)  # This also shows the cpf plot.
