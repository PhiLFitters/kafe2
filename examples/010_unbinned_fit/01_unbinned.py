#!/usr/bin/env python
"""
kafe2 example: Unbinned Fit
================================

An unbinned fit is needed, when there are too few data points to create a (good) histogram.
If a histogram is created from too few data points, information can be lost or even changed,
by changing the exact value of one data point to the range of a bin.
With an unbinned likelihood fit it's still possible to fit the PDF to the data points, as the
likelihood of each data point is fitted.

In this example the decay time of muons is calculated. The data was collected using the Kamiokanne
experiment (a water-Cherenkov detector with photomultiplier readout). For more information on the
experimental setup visit https://github.com/GuenterQuast/picoCosmo.
A visual representation of the data can be found here:
https://github.com/GuenterQuast/picoCosmo/blob/master/doc/dpFigs_Kanne.pdf
"""

from kafe2.fit import UnbinnedContainer, Fit, Plot
from kafe2.fit.tools import ContoursProfiler

import numpy as np


def pdf(t, tau=2.2, fbg=0.1, a=1.0, b=9.75):
    """
    Probability density function for the decay time of a muon using the Kamiokanne-Experiment.
    The PDF is normed for the interval (a, b).

    :param t: decay time
    :param fbg: background
    :param tau: expected mean of the decay time
    :param a: the minimum decay time which can be measured
    :param b: the maximum decay time which can be measured
    :return: probability for decay time x
    """
    pdf1 = np.exp(-t / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
    pdf2 = 1. / (b - a)
    return (1 - fbg) * pdf1 + fbg * pdf2


# load the data from the experiment
infile = "tau_mu.dat"
dT = np.loadtxt(infile)

data = UnbinnedContainer(dT)  # create the kafe data object
data.label = 'lifetime measurements'
data.axis_labels = ['life time $\\tau$ (µs)', 'Density']

# create the fit object and set the pdf for the fit
fit = Fit(data=data, model_function=pdf)

# Fix the parameters a and b.
# Those are responsible for the normalization of the pdf for the range (a, b).
fit.fix_parameter("a", 1)
fit.fix_parameter("b", 11.5)
# constrain parameter fbg to avoid unphysical region
fit.limit_parameter("fbg", 0., 1.)

# assign latex names for the parameters for nicer display
fit.model_label = "exponential decay law + flat background"
fit.assign_parameter_latex_names(fbg='f')
# assign a latex expression for the fit function for nicer display
fit.assign_model_function_latex_expression("\\frac{{ (1-{fbg}) \\, e^{{-{t}/{tau}}} }}"
                                           "{{{tau} \\, (e^{{-{a}/{tau}}}-e^{{-{b}/{tau}}}) }}"
                                           "+ \\frac{{ {fbg}}} {{{b}-{a}}}")

fit.do_fit()  # perform the fit
fit.report(asymmetric_parameter_errors=True)  # print a fit report to the terminal

plot = Plot(fit)  # create a plot object
plot.plot(fit_info=True, asymmetric_parameter_errors=True)  # plot the data and the fit

# Optional: create a contours profile
cpf = ContoursProfiler(fit, profile_subtract_min=False)
# Optional: plot the contour matrix for tau and fbg
cpf.plot_profiles_contours_matrix(parameters=['tau', 'fbg'])

plot.show()  # show the plot(s)
