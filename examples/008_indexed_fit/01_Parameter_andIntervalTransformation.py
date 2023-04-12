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
from kafe2 import indexed_fit, plot

# example of parameters: (r, phi) of a space point in polar coordinates
pars = np.array([0.9, 0.755])
puncs = np.array([0.01, 0.15])

# define a function for the transformation to Cartesian
def cartesian_to_polar(x, y):

    # access data container to find out how many measurements we got
    nm = len(pars)//2 # expect 2 concatenated arrays (r and phi)

    # determine polar coordinates from Cartesian (x,y)
    r = np.ones(nm)*np.sqrt(x*x + y*y)
    phi = np.ones(nm)*np.arctan2(y, x)

    return np.concatenate( (r, phi) )

# Call kafe2 wrapper function to do an indexed fit:
indexed_fit(model_function=cartesian_to_polar, data=pars, error=puncs)
plot(
    data_label="Measured coordinates",
    model_label=r'$r$-$\phi$ from x-y',
    x_label=r"Index 0=$r$, 1=$\phi$",
    y_label="Measured Coordinates"
)
