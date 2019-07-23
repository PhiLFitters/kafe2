from kafe2 import UnbinnedContainer, UnbinnedFit
import numpy as np
import matplotlib.pyplot as plt


def pdf(x, fbg=0.1, tau=2.2):
    """
    Probability density function for the decay time of a myon
    :param x: decay time
    :param fbg: background
    :param tau: expected decay time
    :return: probability for decay time t
    """
    pdf1 = np.exp(-x / tau) / tau / (np.exp(-1 / tau) - np.exp(-9.75 / tau))
    pdf2 = 1./(9.75-1)
    return (1-fbg)*pdf1 + fbg*pdf2


infile = "tau_mu.dat"

dT = np.loadtxt(infile)

data = UnbinnedContainer(dT)
fit = UnbinnedFit(data=data, model_density_function=pdf)

fit.do_fit()
