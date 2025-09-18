#!/usr/bin/env python

"""
kafe2 example: Histogram Fit (Pitfalls)
=======================================

This example demonstrates why it is more convenient to use the HistFit class
instead of XYFit when dealing with histogrammed data.

While it is in principle possible to perform such fits correctly using XYFit,
it requires much more care. This example shows common mistakes that can occur
when that necessary care is not taken and how this makes the fit results worse.

We will especially look at two scenarios that typically arise from having only small amounts of data.
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

Another problem in the code above is the use of incorrect y-values for the fit.
Naively, the y-value of the model function at the position of the bin midpoints was used.
In fact, since probability densities and histogrammed data are being treated, the correct way
would be to integrate the model function over each bin to obtain the correct expected count of events in the bin.
It can be seen in the following that even if we combine bins in order to get rid of the empty bin,
the fit will still not yield perfect results.

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

'''
Now you should see that the fit will at least converge, but the results have large uncertainties.
This is, among other things, because we still don't use the integral over our bins as the y-value for our model in the fit.
Next, the HistContainer, which automatically triggers kafe2 to use the HistFit class for fitting, will be used.
This will yield the best results without any further effort, even without eliminating the empty bin beforehand.
'''

#We will use our initial bincounts and binning
bincounts = np.array([1, 0, 3, 9, 7, 8, 1, 1])
binedges = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

hist_data = HistContainer(bin_edges=binedges)
hist_data.set_bins(bincounts)

#This is everything we have to prepare befor performing the fit
Histfit = Fit(hist_data, model_function=normal_distribution)
Histfit.do_fit()

Plot3 = Plot(Histfit)
Plot3.plot()
plt.show()




