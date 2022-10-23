#!/usr/bin/env python

"""
kafe2 example: Histogram Fit (Absolute)
=======================================

This example is equivalent to the other histogram example in this folder except for the fact that
the model function is not a density but has an amplitude as one of its parameters. Long-term this
will be replaced by a better example.
"""

import numpy as np
from kafe2 import HistContainer, Fit, Plot


def normal_distribution(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)


# random dataset of 100 random values, following a normal distribution with mu=0 and sigma=1
data = np.random.normal(loc=0, scale=1, size=100)

# Create a histogram from the dataset by specifying the bin range and the amount of bins.
# Alternatively the bin edges can be set.
histogram = HistContainer(n_bins=10, bin_range=(-5, 5), fill_data=data)

# create the Fit object by specifying a scalable model function
fit = Fit(data=histogram, model_function=normal_distribution, density=False)

fit.do_fit()  # do the fit
fit.report(asymmetric_parameter_errors=True)  # Optional: print a report to the terminal

# Optional: create a plot and show it
plot = Plot(fit)
plot.plot(asymmetric_parameter_errors=True)
plot.show()
