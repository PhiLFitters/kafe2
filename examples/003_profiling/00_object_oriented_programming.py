#!/usr/bin/env python
"""
kafe2 example: Object-Oriented Programming
==========================================

Until now fits were performed using the methods kafe2.xy_fit and kafe2.plot. These methods provide
pre-configured pipelines for fitting models to xy data and plotting the results. Internally they use
objects (in the programming sense) which represent things like data or fits as a whole. From now on
the examples will use the objects directly. This is slightly more complicated but it also provides
greater flexibility.

This example serves as an introduction to kafe2 objects. The implementation is mostly comparable to
001_line_fit/line_fit.py
"""

from kafe2 import XYContainer, Fit, Plot, ContoursProfiler

# Create an XYContainer object to hold the xy data for the fit:
xy_data = XYContainer(x_data=[1.0, 2.0, 3.0, 4.0],
                      y_data=[2.3, 4.2, 7.5, 9.4])
# x_data and y_data are combined depending on their order.
# The above translates to the points (1.0, 2.3), (2.0, 4.2), (3.0, 7.5), and (4.0, 9.4).

# Important: Specify uncertainties for the data:
xy_data.add_error(axis='x', err_val=0.1)
xy_data.add_error(axis='y', err_val=0.4)

xy_data.label = 'Data'  # How the data is called in plots

# Create an XYFit object from the xy data container.
# By default, a linear function f=a*x+b will be used as the model function.
line_fit = Fit(data=xy_data)

# Perform the fit: Find values for a and b that minimize the
#     difference between the model function and the data.
line_fit.do_fit()  # This will throw a warning if no errors were specified.

# Optional: Print out a report on the fit results on the console.
line_fit.report()
# With kafe2.xy_fit this gets printed to a file.

# Optional: Create a plot of the fit results using Plot.
plot = Plot(fit_objects=line_fit)  # Create a kafe2 plot object.
plot.x_label = 'x'  # Set x axis label.
plot.y_label = 'y'  # Set y axis label.
plot.plot()  # Do the plot.

plot.save()  # Saves the plot to file 'fit.png' .
# plot.save('my_fit.pdf')  # Saves the plot to a different file / with a different file extension.

# Use the ContoursProfiler object to create contour plots:
cpf = ContoursProfiler(line_fit)
cpf.plot_profiles_contours_matrix()

# Show the fit result.
plot.show()  # Just a convenience wrapper for matplotlib.pyplot.show() .
# NOTE: Calling matplotlib.pyplot.show() closes all figures by default so call this AFTER saving.

# Alternatively you could still use the function kafe.plot like this:
# from kafe2 import plot  # Notice that "plot" is not capitalized. "Plot" is the plot object.
# kafe2.plot(line_fit)  # Pass the fit object to the plot function.
