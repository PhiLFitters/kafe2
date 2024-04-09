#!/usr/bin/env python
"""
kafe2 binned Histogram Fit: signal on flat background
=====================================================

Fitting of models to histograms should generally be done using a "binned likelihood fit", 
which takes into account the Poisson nature of the uncertainties on the bin entries.

kafe2 comfortably provides such functionality via the classes HistContainer and
HistFit. The "fit function" must be provided as a normalized probability density
as the default. Normalizing to the number of entries in the histrogram is also
possible by setting the option *density=False*; in such cases, an additinal
parameter must be provided in the fit function to take care of the normalization.

The example below is a minimalist one to determines the fraction of a gaussian-shaped
signal on top of a flat background and may be used a the basis for any type of signal
shape on a backgound distribution.

Some notes:

  - The *kafe2* class *HistContainer* provides various methods to create a histogram 
    from input data. It accepts raw data, numpy historgrams or numpy arrays 
    to directly set the bin contents
  - The bin contents according to the model is determined by numerical integration
    of the pdf from the left to the right bin edges. 
  - The quality of fit, labeled as "-2ln L_R", is based on the ratio of the maximized 
    likelihood of the data w.r.t the model function and the so-called "fully saturated 
    model", which, in case of a histogram, represents a histogram exactly matching the 
    bin entries observed in the data. This quanity becomes equal to the usual Chi2 value 
    in case of sufficiently large numbers of entries per bin.     
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistContainer, Fit, Plot

# parameters of data sample, signal and background parameters
N = 200  # number of entries
min = 0.0  # range of data, minimum
max = 10.0  # maximum
s = 0.8  # signal fraction
pos = 6.66  # signal position
width = 0.33  # signal width


def generate_data(N=100, min=0, max=1.0, pos=0.0, width=0.25, signal_fraction=0.1):
    """generate a random dataset:
    Gaussian signal at position p with width w and signal fraction s
    on top of a flat background between min and max
    """
    # signal sample
    data_s = np.random.normal(loc=pos, scale=width, size=int(signal_fraction * N))
    # background sample
    data_b = np.random.uniform(low=min, high=max, size=int((1 - signal_fraction) * N))
    return np.concatenate((data_s, data_b))

def s_plus_bPDF(x, mu=3.0, sigma=2.0, s=0.5):
    """(normalized) pdf of a Gaussian signal on top of flat background"""
    normal = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma**2)
    flat = 1.0 / (max - min)
    return s * normal + (1 - s) * flat

def s_plus_b(x, Ns = 200, mu=3.0, sigma=2.0, Nb = 50.):
    """Gaussian signal on top of flat background"""
    normal = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma**2)
    flat = 1.0 / (max - min)
    return Ns * normal + Nb * flat


if __name__=="__main__": # -----------------------------

  # generate a histogram data sample
  SplusB_data = generate_data(N, min, max, pos, width, s)
  # show the histogram
  # bc, be, _ = plt.hist(SplusB_data, bins=35, rwidth=0.9)

  # Create a histogram Container from the dataset
  SplusB_histcontainer = HistContainer(n_bins=35, bin_range=(min, max), fill_data=SplusB_data)
  SplusB_histcontainer.label = "artifical data"

  # create the Fit object by specifying a density function
  hist_PDFfit = Fit(data=SplusB_histcontainer, model_function=s_plus_bPDF)

  # as an alternative, specify unnormalized fit function
  hist_fit = Fit(data=SplusB_histcontainer, density=False,
                 model_function=s_plus_b)
  
  hist_PDFfit.do_fit()  # 1st fit
  hist_PDFfit.report()  # optional: print a report to the terminal

  hist_fit.do_fit()  # 2nd fit
  hist_fit.report()  # optional: print a report to the terminal

  # Optional: create plot and show it
  hist_plot = Plot([hist_PDFfit, hist_fit])
  hist_plot.plot(asymmetric_parameter_errors=True)
  plt.show()
