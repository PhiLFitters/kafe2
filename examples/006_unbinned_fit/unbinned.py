from kafe2 import UnbinnedContainer, UnbinnedFit, ContoursProfiler
import numpy as np
import matplotlib.pyplot as plt


def pdf(x, tau=2.2, fbg=0.1):
    """
    Probability density function for the decay time of a myon
    :param x: decay time
    :param fbg: background
    :param tau: expected decay time
    :return: probability for decay time t
    """
    b = 11.5
    a = 1.
    pdf1 = np.exp(-x / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
    pdf2 = 1. / (b - a)
    return (1 - fbg) * pdf1 + fbg * pdf2


infile = "tau_mu.dat"

dT = np.loadtxt(infile)

data = UnbinnedContainer(dT)
fit = UnbinnedFit(data=data, model_density_function=pdf)

fit.do_fit()
fit.report()

cpf = ContoursProfiler(fit, profile_subtract_min=False)
cpf.plot_profiles_contours_matrix(parameters=['tau'],
                                  show_grid_for='all',
                                  show_fit_minimum_for='all',
                                  show_error_span_profiles=True,
                                  show_legend=True,
                                  show_parabolic_profiles=True,
                                  show_ticks_for='all',
                                  contour_naming_convention='sigma',
                                  label_ticks_in_sigma=False)
