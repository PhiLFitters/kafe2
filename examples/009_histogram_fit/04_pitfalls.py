#!/usr/bin/env python

"""
kafe2 example: Histogram Fit (Pitfalls)
=======================================

This example demonstrates why it is more convenient to use the HistFit class
instead of XYFit when dealing with histogrammed data.

While it is in principle possible to perform such fits correctly using XYFit,
it requires much more care. This example shows common mistakes that can occur
when that necessary care is not taken and how this makes the fit results worse.

We will especially look at two scenarios that can arise from having only small amounts of data.
This is, for example, a problem in the search for new rare processes in high-energy physics.
Assume we are looking for a Gaussian signal peak, free from background for simplicity
(the treatment of a signal over background is explained in example 03_SpluBfit.py).

The first problem arises if zero events are present in a bin. Since we assume Poisson uncertainties
on the data points, an empty bin will be assumed to have an uncertainty of zero when using a Gaussian approximation.
By default, XYFit uses a χ² cost function, which describes data with Gaussian uncertainties.
This means empty bins will be assumed to be known with perfect precision, and
the model function is forced to pass this point exactly. Of course, this only happens when no other (systematic)
uncertainties are given. It can be seen in the following example script how this makes an interpretable
result impossible. Since the HistFit class uses a Poisson likelihood by default, this first problem is avoided.
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYContainer, Fit, HistContainer, Plot

def normal_distribution(x, mu = 0, sigma = 1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)

#Manually specify already binned data, to provoke empty bins (generated uniformly with mu = 0, sigma = 1)
bincounts = np.array([1, 0, 3, 9, 7, 8, 1, 1])
binedges = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

#Now naively perform a default XYFit
x_data = binmids = np.mean([binedges[:-1], binedges[1:]], axis=0) #use binmids as x values
y_data = bincounts/(np.sum(bincounts)*np.diff(binedges))
print(y_data) #use normalized histogram as y data
x_error = 0.25 #use half binwidht as x_error
y_error = np.sqrt(y_data)

xy_data = XYContainer(x_data=x_data, y_data=y_data)
xy_data.add_error(err_val=x_error, axis="x")
xy_data.add_error(err_val=y_error, axis="y")

XYFit_1 = Fit(xy_data, normal_distribution)
XYFit_1.do_fit()
#create a plot
Plot1 = Plot(XYFit_1)
Plot1.plot()
plt.show()



"""
This first fit should give you warnings about the cost function being evaluated as infinite, which comes from the empty bin.
Also, the result of the fit should not return any good values. This is the result of the empty bin.

It can be seen in the following that even if we combine bins in order to get rid of the empty bin,
the fit will converge and not throw errors, but still not yield perfect results.
"""


bincounts = np.array([1, 3, 9, 7, 8, 2]) 
binedges = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])

#Now again perform a default XYFit
x_data = binmids = np.mean([binedges[:-1], binedges[1:]], axis=0) #use binmids as x values
y_data = bincounts/(np.sum(bincounts)*np.diff(binedges)) #use normalized histogram as y data
x_error = np.diff(binedges)/2 #use half binwidht as x_error
y_error = np.sqrt(y_data)

xy_data = XYContainer(x_data=x_data, y_data=y_data)
xy_data.add_error(err_val=x_error, axis="x")
xy_data.add_error(err_val=y_error, axis="y")

XYFit_2 = Fit(xy_data, normal_distribution)
XYFit_2.do_fit()
#create a plot
Plot2 = Plot(XYFit_2)
Plot2.plot()
plt.show()



"""
In the code above, the uncertainties on each bin were simply specified as the square root of the counts in each bin.
In reality, the measured bin counts are just one random experiment. It is the outcome of a random draw
from a Poisson distribution. The mean, and thus also the Poisson uncertainty of these bin counts, stems from the
expected number of events in that bin given by the model function. That means calculating the uncertainties like
it was done above introduces a bias. If there are more events than expected in a bin, the uncertainties are overestimated.
On the other hand, they are underestimated if the observed number of events in a bin is lower than expected.

This problem can be reduced by doing a so-called prefit. The idea is to use the fit from above as a first approximation
to our true distribution. Then uncertainties are obtained from the expected number of events in each bin as predicted
by the prefit. 
"""

bincounts = np.array([1, 3, 9, 7, 8, 2]) 
binedges = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])

#Now again perform a default XYFit
x_data = binmids = np.mean([binedges[:-1], binedges[1:]], axis=0) #use binmids as x values
y_data = bincounts/(np.sum(bincounts)*np.diff(binedges)) #use normalized histogram as y data
x_error = np.diff(binedges)/2 #use half binwidht as x_error

prefit_result = XYFit_2.parameter_values
y_error_model = np.sqrt(normal_distribution(binmids, *prefit_result))#Now instead of the measured data, use the prefit to give us the expected bincounts

#print both y_errors, from the prefit and from this fit for comparison
print("The y errors used in the prefit, relative to the datapoints are: " + str(y_error))
print("The y errors used now, relative to the model with parameters from the prefit: " + str(y_error_model))


xy_data = XYContainer(x_data=x_data, y_data=y_data)
xy_data.add_error(err_val=x_error, axis="x")
xy_data.add_error(err_val=y_error_model, axis="y")

XYFit_3 = Fit(xy_data, normal_distribution)
XYFit_3.do_fit()
#create a plot
Plot3 = Plot(XYFit_3)
Plot3.plot()
plt.show()

"""
The uncertainties on the result were already reduced slightly in comparison to the prefit. And by looking at the printed
y-errors for the prefit and the final fit, differences up to 20% can be observed.
Still the uncertainties are larger than they should be. This is also due to one final problem that
is discussed in this example script.

One problem in all the code above is the use of incorrect y-values for the fit.
Naively, the y-value of the model function at the position of the bin midpoints was used.
In fact, since probability densities and histogrammed data are being treated, the correct way
would be to integrate the model function over each bin to obtain the correct expected count of events in the bin.
This will not be shown here.

We can see that the result of our fit gets better with every improvement we do to our fitting procedure.
But a lot of effort has to be put in to get reasonable results.
Next, the HistContainer, which automatically triggers kafe2 to use the HistFit class for fitting, will be used.
This will yield the best results without any further effort, even without eliminating the empty bin beforehand.
Furthermore, the Plot shows nicely the model function prediction for the histogram.
"""

#We will use our initial bincounts and binning
bincounts = np.array([1, 0, 3, 9, 7, 8, 1, 1])
binedges = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

hist_data = HistContainer(bin_edges=binedges)
hist_data.set_bins(bincounts)

#This is everything we have to prepare befor performing the fit
Histfit = Fit(hist_data, model_function=normal_distribution)
Histfit.do_fit()

Plot4 = Plot(Histfit)
Plot4.plot()
plt.show()




