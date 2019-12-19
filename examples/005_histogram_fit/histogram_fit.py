#!/usr/bin/env python

"""
kafe2 example: Histogram Fit
=======================

kafe2 is not only capable of performing XY-Fits. This example shows how to handle a one dimensional dataset.
The distribution of a random stochastic variable follows a probability density function. This density needs to be
normalized. The fit will determine the parameters of that density function, which the dataset is most likely to follow.
If the given function is not normalized, please make sure to add an independent scale factor. The hist fit will try to
normalize the function to one for the bin range.
To get to the height of a bin, please multiply the results of the fitted function with the amount of entries N of the
histogram.
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistContainer, HistFit, Plot


def normal_distribution_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)


# create a random dataset of 100 random values, following a normal distribution with mu=0 and sigma=1
data = np.random.normal(loc=0, scale=1, size=100)

# create a histogram from the dataset by specifying the bin range and the amount of bins
# alternatively the bin edges can be set
histogram = HistContainer(n_bins=10, bin_range=(-5, 5), fill_data=data)

# create the Fit object by specifying a density function
fit = HistFit(data=histogram, model_density_function=normal_distribution_pdf)

fit.do_fit()  # do the fit
fit.report()  # Optional: print a report to the terminal

# Optional: create a plot and show it
plot = Plot(fit)
plot.plot()
plt.show()
