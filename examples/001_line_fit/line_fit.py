#!/usr/bin/env python
"""
kafe2 example: Line Fit
=======================

The simplest, and also the most common use case of a fitting framework
is performing a line fit: A linear function of the form
f(x) = a * x + b is made to align with a series of xy data points that
have some uncertainty along the x-axis and the y-axis.
This example demonstrates how to perform such a line fit in kafe2 and
how to extract the results.
"""

from kafe2 import XYContainer, XYFit, Plot
import matplotlib.pyplot as plt

# Create an XYContainer object to hold the xy data for the fit.
xy_data = XYContainer(x_data=[1.0, 2.0, 3.0, 4.0],
                      y_data=[2.3, 4.2, 7.5, 9.4])
# x_data and y_data are combined depending on their order.
# The above translates to the points (1.0, 2.3), (2.0, 4.2), and (4.0, 9.4).

# Important: Specify uncertainties for the data.
xy_data.add_error(axis='x', err_val=0.1)
xy_data.add_error(axis='y', err_val=0.4)

# Create an XYFit object from the xy data container.
# By default, a linear function f=a*x+b will be used as the model function.
line_fit = XYFit(xy_data=xy_data)

# Perform the fit: Find values for a and b that minimize the
#     difference between the model function and the data.
line_fit.do_fit()  # This will throw an exception if no errors were specified.

# Optional: Print out a report on the fit results on the console.
line_fit.report()

# Optional: Create a plot of the fit results using Plot.
plot = Plot(fit_objects=line_fit)  # Create a kafe2 plot object.
plot.plot()  # Do the plot.

# Show the fit result.
plt.show()
