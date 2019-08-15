#!/usr/bin/env python2
"""
kafe2 example: Unbinned Fit
================================

An unbinned fit is needed, when there are too few data points to create
a histogram which follows the pdf one wants to fit.
With an unbinned likelihood fit it's still possible to fit the pdf to the
data points.
In this example we want to calculate the decay time of a myon. As there are
only a few events in the detector, we use an unbinned fit.
"""

from kafe2 import UnbinnedContainer, UnbinnedFit, ContoursProfiler, UnbinnedPlot
import numpy as np
import matplotlib.pyplot as plt


def pdf(x, tau=2.2, fbg=0.1, a=1., b=9.75):
    """
    Probability density function for the decay time of a myon
    :param x: decay time
    :param fbg: background
    :param tau: expected decay time
    :param a: the minimum decay time which can be measured
    :param b: the maximum decay time which can be measured
    :return: probability for decay time t
    """
    pdf1 = np.exp(-x / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
    pdf2 = 1. / (b - a)
    return (1 - fbg) * pdf1 + fbg * pdf2


infile = "tau_mu.dat"

dT = np.loadtxt(infile)

data = UnbinnedContainer(dT)
fit = UnbinnedFit(data=data, model_density_function=pdf)

fit.fix_parameter("a", 1)
fit.fix_parameter("b", 11.5)

fit.assign_parameter_latex_names(tau=r'\tau', fbg='f', a='a', b='b')
fit.assign_model_function_latex_expression("(1-{fbg}) \\frac{{e^{{-{x}/{tau}}}}}"
                                           "{{{tau}(e^{{-{a}/{tau}}}-e^{{-{b}/{tau}}})}}"
                                           "+ {fbg} \\frac{{1}}{{{b}-{a}}}")

fit.do_fit()
fit.report()

plot = UnbinnedPlot(fit)
plot.plot()
plot.show_fit_info_box()  # Optional: Add numerical fit results to the image.
cpf = ContoursProfiler(fit, profile_subtract_min=False)
cpf.plot_profiles_contours_matrix(parameters=['tau', 'fbg'])
plt.show()
