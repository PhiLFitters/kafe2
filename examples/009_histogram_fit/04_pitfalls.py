#!/usr/bin/env python

"""
kafe2 example: Histogram Fit (Pitfalls)
=======================================

This example demonstrates a scenario in which it is more convenient to use the
HistFit class rather than the XYFit class.

While it is in principle possible to perform such fits correctly using XYFit,
more care must be taken to avoid unusable or biased fit results.
This example shows common problems that can occur when using XYFit and how to fix them.
HistFit handles these problems automatically.

We will try to build the equivalent of the kafe2 HistFit by hand using an XYFit.
In general histogram Fits are useful when handling large amounts of data.
The computational complexity of e.g. the chi2 cost function scales with the number of inputs cubed
(when considering correlations). So reducing the raw data to a comparatively
small number of bins is crucial to make the fit computationally feasible.

The default XYFit has some problems that have to be addressed, when histogrammed data is
processed. The XYFit does not have any built-in functionality to transform raw data into
histogrammed data. This means if we want to use histogrammed data, this has to be done manually.
In the following, 30 datapoints, sampled from a normal distribution, are used.
"""
import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYContainer, Fit, HistContainer, Plot


def normal_distribution(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma**2)


# Fill the datapoints into bins (generated uniformly with mu = 0, sigma = 1)
data = np.array(
    [
        0.15452608,
        0.15814163,
        1.84110142,
        -0.27350108,
        -1.89289518,
        0.21379812,
        -0.27967822,
        0.21451591,
        -0.07221073,
        -0.51282405,
        0.39306818,
        1.12542323,
        -0.71070046,
        1.86784967,
        -0.17510427,
        1.72786353,
        0.95243581,
        -0.71476994,
        -0.0816407,
        -0.38590102,
        0.92506023,
        0.4333063,
        0.86662424,
        -0.78075927,
        1.36575012,
        -0.49828474,
        0.14649399,
        1.40194082,
        -1.50842127,
        1.21646781,
    ]
)

bin_edges = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
bin_counts, bin_edges = np.histogram(data, bins=bin_edges, density=True)

# Now naively perform a default XYFit
x_data = bincenters = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)  # use bincenters as x values
y_data = bin_counts  # use normalized histogram as y data
y_error = np.sqrt(bin_counts) / (np.sum(bin_counts) * np.diff(bin_edges))  # assume Poisson errors on our bin_counts

xy_data = XYContainer(x_data=x_data, y_data=y_data)
xy_data.add_error(err_val=y_error, axis="y")

XYFit_1 = Fit(xy_data, normal_distribution)
XYFit_1.do_fit()
# create a plot
Plot1 = Plot(XYFit_1)
Plot1.plot()
plt.show()
"""
When performing the fit, errors like "The cost function was evaluated as infinite" appear in the output. 
Furthermore, when looking at the plot of our fit result, it is clear that this fit didn't return the result we expected.
The starting values for the fit are returned as best fit value, and the uncertainties are reported as NaN.
What happened? The problem arises because Poisson uncertainties were assumed for the bin counts.
If a bin is empty, the uncertainty is treated as zero by the fit. Thus the model function is forced to pass this datapoint
exactly or the cost function will be infinite.
This occures because the XYFit uses a χ²-cost function by default, which is only valid 
for Gaussian uncertainties, but not in the case of Poisson uncertainties. To fix this, the
cost function of the fit is changed to a Poisson negative log-likelhoood (NLL).
"""


# rescale normal distribution by bin counts and bin width
def normal_distribution_scaled(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma**2) * np.sum(bin_counts) * 0.5


bin_counts, bin_edges = np.histogram(data, bins=bin_edges)

x_data = bincenters = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)  # use bincenters as x values
y_data = bin_counts  # use bin_counts as y data (not normalized now)

xy_data = XYContainer(x_data=x_data, y_data=y_data)

XYFit_2 = Fit(xy_data, normal_distribution_scaled, cost_function="poisson")
XYFit_2.do_fit()
# create a plot
Plot2 = Plot(XYFit_2)
Plot2.plot()
plt.show()

"""
Now it wasn't even necessary to explicitly specify the y-errors. By using a Poisson NLL,
the y-errors are no longer calculated from the measured bin counts, but instead from the model expectation. 
This handles empty bins correctly and also prevents getting biased uncertainties.

Another subtlety is the definition of the y_data: So far, simply the midpoint of each bin was used. This is only
a linear approximation of the behaviour of the model function between the bin edges. The HistFit class of kafe2
on the other hand uses "Simpson's rule", a method to approximate the behaviour quadratically for more accuracy.
The most accurate albeit computationally expensive method would be to integrate the model function over each bin.

The implementation of Simpson's rule in our procedure using a XYFit will not be done here,
since the influence is rather small in this case. Instead, it is shown how it is much easier to just use the HistFit
class of kafe2. The binning is automatically done by the HistContainer, the Poisson NLL is the default cost function
and the Simpson rule is already implemented. 
"""
# We will use our initial data and binning
hist_data = HistContainer(bin_edges=bin_edges, fill_data=data)


# This is everything we have to prepare befor performing the fit
Histfit = Fit(hist_data, model_function=normal_distribution, density=True)
Histfit.do_fit()

Plot3 = Plot(Histfit)
Plot3.plot()
plt.show()

"""
Now consider the case, where systematical errors next to the statistical ones are present. Suppose each bin has an additional
Gaussian uncertainty of 1. Now there are two different types of uncertainties in the Fit, that can't be simply added.
However, due to the central limit theorem, for sufficiently large event counts the Poisson distribution approaches
a normal distrbution. Therefore, by switching the cost function from a Poisson NLL to the Gaussian approximation,
the fit can be performed correctly again. Note that this gaussian approximation is different to the one that was falsly used
in the beginning of the example. Here statistical uncertainties are calculated from the model function, while above,
they were calculated from the measured data.
"""
# We will use our initial data and binning
hist_data = HistContainer(bin_edges=bin_edges, fill_data=data)
hist_data.add_error(err_val=1)

# This is everything we have to prepare befor performing the fit
Histfit = Fit(hist_data, model_function=normal_distribution, density=True, cost_function="gauss-approximation")
Histfit.do_fit()

Plot4 = Plot(Histfit)
Plot4.plot()
plt.show()
