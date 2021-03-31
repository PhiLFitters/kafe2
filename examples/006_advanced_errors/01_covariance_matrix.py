#!/usr/bin/env python
"""
kafe2 example: Weighted average of measurements
===============================================
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import IndexedFit, Plot


##################
# Model function #
##################
def W_mass_model(m_W=80):
    # our model is a simple constant function
    #return float(m_W)  # FIXME: automatically adapt to measurement list length
    return [float(m_W)]*8


# measurements of the W boson mass
W_mass_measurements = [80.429, 80.339, 80.217, 80.449, 80.477, 80.310, 80.324, 80.353]
# each measurement has a statistical uncertainty
W_mass_stat_err =     [0.055,   0.073,  0.068,  0.058,  0.069,  0.091,  0.078,  0.068]
#W_mass_stat_err =     [1,       1,      1,      1,      1,      1,      1,      1    ]

# construct a covariance matrix for the systematic errors
n_measurements = len(W_mass_measurements)
W_mass_cov_mat = np.matrix(np.zeros((n_measurements, n_measurements)))
W_mass_cov_mat[:4, :4] = 0.021**2
W_mass_cov_mat[4:, 4:] = 0.044**2
W_mass_cov_mat[:4, 4:] = 0.025**2
W_mass_cov_mat[4:, :4] = 0.025**2


# create an IndexedFit, specifying the measurement data, model function and cost function
f = IndexedFit(data=W_mass_measurements,
               model_function=W_mass_model)

# add two error sources
f.add_error(W_mass_stat_err, correlation=0)    # statistical errors
f.add_matrix_error(W_mass_cov_mat, matrix_type='cov') # systematic errors (using a covariance matrix)

# assign LaTeX strings to various quantities (for nice display)
f.assign_parameter_latex_names(m_W=r"m_{\rm W}")  # the function parameters
f.assign_model_function_latex_expression(r"{0}")  # the function expression

#f._nexus.print_state()

# perform the fit
f.do_fit()

# print out information about the fit
f.report()

# to see the fit results, plot using Plot
p = Plot(f)
p.plot(fit_info=True)

# show the fit result
plt.show()
