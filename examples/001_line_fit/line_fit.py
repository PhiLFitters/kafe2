#!/usr/bin/env python
"""
kafe2 example: Line Fit
=======================

The simplest, and also the most common use case of a fitting framework
is performing a line fit: A linear function of the form
f(x) = a * x + b is made to align with a series of xy data points that
have some uncertainty along the x axis and the y axis.
This example demonstrates how to perform such a line fit in kafe2 and
how to extract the results.
"""

import kafe2

# Define or read in the data for your fit:
x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.3, 4.2, 7.5, 9.4]
# x_data and y_data are combined depending on their order.
# The above translates to the points (1.0, 2.3), (2.0, 4.2), (3.0, 7.5), and (4.0, 9.4).

# Important: Specify uncertainties for the data!
x_error = 0.1
y_error = 0.4

# Pass the information to kafe2:
kafe2.xy_fit(x_data, y_data, x_error=x_error, y_error=y_error)
# Because no model function was specified a line is used by default.

# Call another function to create a plot:
kafe2.plot(
    x_label="x",  # x axis label
    y_label="y",  # y axis label
    data_label="Data",  # label of data in legend
)
