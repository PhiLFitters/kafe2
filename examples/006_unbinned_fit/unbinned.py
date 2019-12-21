#!/usr/bin/env python
"""
kafe2 example: Unbinned Fit
================================

An unbinned fit is needed, when there are too few data points to create a (good) histogram. If a histogram is created
from too few data points, information can be lost or even changed, by changing the exact value of one data point to the
range of a bin. With an unbinned likelihood fit it's still possible to fit the pdf to the data points, as the likelihood
of each data point is fitted.

In this example the decay time of muons is calculated. Tha data was collected using the Kamiokanne experiment (a
water-Cherenkov detector with photomultiplier readout). For more information on the experimental setup visit
https://github.com/GuenterQuast/picoCosmo. A visual representation of the data can be found here:
https://github.com/GuenterQuast/picoCosmo/blob/master/doc/dpFigs_Kanne.pdf
"""

from kafe2.fit import UnbinnedContainer, UnbinnedFit, Plot
from kafe2.fit.tools import ContoursProfiler

import numpy as np
import matplotlib.pyplot as plt


def pdf(x, tau=2.2, fbg=0.1, a=1., b=9.75):
    """
    Probability density function for the decay time of a myon using the Kamiokanne-Experiment. The pdf is normed for the
    interval (a, b).

    :param x: decay time
    :param fbg: background
    :param tau: expected mean of the decay time
    :param a: the minimum decay time which can be measured
    :param b: the maximum decay time which can be measured
    :return: probability for decay time x
    """
    pdf1 = np.exp(-x / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
    pdf2 = 1. / (b - a)
    return (1 - fbg) * pdf1 + fbg * pdf2


# load the data from the experiment
infile = "tau_mu.dat"
dT = np.loadtxt(infile)

data = UnbinnedContainer(dT)  # create the kafe data object
fit = UnbinnedFit(data=data, model_density_function=pdf)  # create the fit object and set the pdf for the fit

# Fix the parameters a and b. Those are responsible for the normalization of the pdf for the range (a, b).
fit.fix_parameter("a", 1)
fit.fix_parameter("b", 11.5)

# assign latex names for the parameters for nicer display
fit.assign_parameter_latex_names(tau=r'\tau', fbg='f', a='a', b='b')
# assign a latex expression for the fit function for nicer display
fit.assign_model_function_latex_expression("(1-{fbg}) \\frac{{e^{{-{x}/{tau}}}}}"
                                           "{{{tau}(e^{{-{a}/{tau}}}-e^{{-{b}/{tau}}})}}"
                                           "+ {fbg} \\frac{{1}}{{{b}-{a}}}")

fit.do_fit()  # perform the fit
fit.report()  # print a fit report to the terminal

plot = Plot(fit)  # create a plot object
plot.plot(with_fit_info=True)  # plot the data and the fit

cpf = ContoursProfiler(fit, profile_subtract_min=False)  # Optional: create a contours profile
cpf.plot_profiles_contours_matrix(parameters=['tau', 'fbg'])  # Optional: plot the contour matrix for tau and fbg

plt.show()  # show the plot(s)
