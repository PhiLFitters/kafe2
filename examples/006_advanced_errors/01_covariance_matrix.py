#!/usr/bin/env python
"""
kafe2 Example: Weighted Average of Measurements
===============================================

A physical parameter is typically investigated through multiple experiments. Because the results of
these experiments are slightly different they can be averaged to receive a more precise estimate of
the physical parameter. However, when this average is calculated it is important to not only
consider the uncertainties of the individual measurements but also whether these uncertainties are
correlated. This can for example happen if two or more measurements are made with the same
experimental setup that leads to the same bias in the result.

In this example we are going to calculate the average of some abstract physical quantity from four
different results. Each result has a statistical uncertainty - these are completely uncorrelated.
The results can further be divided into two distinct groups. Each group has some systematic
uncertainties that affect all measurements of the group in the same way - in other words they are
fully correlated. There are also systematic uncertainties that affect all measurements of both
groups in the same way - these are also fully correlated. To correctly handle the aforementioned
differences in uncertainties we are going to manually construct a covariance matrix which we can
then directly use to specify our uncertainties. We will also see how we could achieve the same
result by specifying multiple simple sources of uncertainty instead.
"""
import numpy as np
from kafe2 import IndexedContainer, Fit, Plot

measurements = np.array([5.3, 5.2, 4.7, 4.8])  # The results we want to average.
err_stat = 0.2  # Statistical uncertainty for each measurement.
err_syst_1234 = 0.15  # Systematic uncertainty that affects all measurements.
err_syst_12 = 0.175  # Systematic absolute uncertainty only affecting the first two measurements.
err_syst_34 = 0.05  # Systematic relative uncertainty affecting only the last two measurements.

# Create an IndexedContainer from our data:
data = IndexedContainer(measurements)

# Start with an empty matrix for our covariance matrix:
covariance_matrix = np.zeros(shape=(4, 4))
# Uncorrelated uncertainties only affect the diagonal of the covariance matrix:
covariance_matrix += np.eye(4) * err_stat ** 2
# Fully correlated uncertainties that affect all measurements affect all covariance matrix entries:
covariance_matrix += err_syst_1234 ** 2
# Fully correlated uncertainties that affect only a part of the measurements result in block-like
# changes to the covariance matrix:
covariance_matrix[0:2, 0:2] += err_syst_12 ** 2  # Magnitude of abs. uncertainty is the same.
err_syst_34_abs = err_syst_34 * measurements[2:4]  # Magnitude of abs. uncertainty is different.
covariance_matrix[2:4, 2:4] += np.outer(err_syst_34_abs, err_syst_34_abs)

# The covariance matrix can now be simply added to our container to specify the uncertainties:
data.add_matrix_error(covariance_matrix, matrix_type="cov")

# To get the same result we could have also added the uncertainties one-by-one like this:
# data.add_error(err_stat, correlation=0)
# data.add_error(err_syst_1234, correlation=1)
# data.add_error([err_syst_12, err_syst_12, 0, 0], correlation=1)
# data.add_error([0, 0, err_syst_34, err_syst_34], correlation=1, relative=True)
# See the next example for another demonstration of this approach.


# Just for graphical output:
data.label = 'Test data'
data.axis_labels = [None, 'Measured value (a.o.)']


# The very simple "model":
def average(a=5.0):
    return a


# Set up the fit:
fit = Fit(data, average)
fit.model_label = 'average value'

# Perform the fit:
fit.do_fit()

# Report and plot results:
fit.report()
p = Plot(fit)
p.plot()

p.show()
