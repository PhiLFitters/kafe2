#!/usr/bin/env python
"""
Example of indexed fit: Transformation of Parameters and Confidence Intervals

The "Data" is a set of parameters to be transformed to some other set
Using a fit, even without any degrees of freedom, allows to transform
confidence contours of the original parameters  to the new parameter 
space.

The example chosen here is very minimalistic:  
A measurement of a space point in 2s space in polar coordinates, 
$r$ and $phi$, is transformed to Cartesian coordinates. The original
Gaussian covariance region in r-phi space transforms to the banana-
shaped region shown in the output. This shape cannot be obtained by
applying the simple (linearized) formula for error propagation. 
"""

import numpy as np, matplotlib.pyplot as plt
from kafe2 import IndexedContainer, Fit, Plot, ContoursProfiler

# example of parameters: (r, phi) of a space point in polar coordinates
pars = np.array([0.9, 0.755])
puncs = np.array([0.01, 0.15])

# Create an IndexedContainer from our data:
indexed_data = IndexedContainer(pars)
indexed_data.add_error(puncs)

# for graphical output:
indexed_data.label = "Measured coordinates"
indexed_data.axis_labels = [r"Index 0=$r$, 1=$\phi$", "Measured coordinates"]

# define a function for the transformation to Cartesian
def cartesian_to_polar(x, y):

    # access data container to find out how many measurements we got
    nm = len(indexed_data.data)//2 # expect 2 concatenated arrays (r and phi)

    # determine polar coordinates from Cartesian (x,y)
    r = np.ones(nm)*np.sqrt(x*x + y*y)
    phi = np.ones(nm)*np.arctan2(y, x)

    return np.concatenate( (r, phi) )

# Set up the fit:
fit = Fit(indexed_data, cartesian_to_polar)
fit.model_label = r'$r$-$\phi$ from x-y'
#fit.assign_model_function_expression("polar")
#fit.assign_model_function_latex_expression("polar")

# Perform the fit:
fit.do_fit()

# Report and plot results:
fit.report(asymmetric_parameter_errors=True)
p = Plot(fit)
p.plot(asymmetric_parameter_errors=True)

contours = ContoursProfiler(fit)
contours.plot_profiles_contours_matrix()

plt.show()
