"""Unbinned log-likelihood fit

   fit a Gaussian signal on a flat background

   compare results from binned and unbinned log-likelihood fits
"""


from kafe2 import Fit, Plot, HistContainer, UnbinnedContainer
import numpy as np
import matplotlib.pyplot as plt


def generate_data(N, min, max, pos, width, s):
    """generate a random dataset:
       Gaussian signal at position p with width w and signal fraction s
       on top of a flat background between min and max
     """
    # signal sample
    data_s = np.random.normal(loc=pos, scale=width, size=int(s*N))
    # background sample
    data_b = np.random.uniform(low=min, high=max, size=int((1-s)*N))
    return np.concatenate((data_s, data_b))


def signal_plus_background(x, mu=3.0, sigma=2.0, s=0.5):
    """pdf of a Gaussian signal on top of flat background
    """
    normal = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)
    flat = 1./(max-min)
    return s * normal + (1-s) * flat


# -- generate a sample of measurements: peak on flat background
N = 200       # number of entries
min = 0.0     # range of data, mimimum
max = 10.0    # maximum
s = 0.8       # signal fraction
pos = 6.66    # signal position
width = 0.33  # signal width
SplusB_data = generate_data(N, min, max, pos, width, s)  


# -- create the kafe data object
unbinned_SplusB = UnbinnedContainer(SplusB_data)  

# -- create the fit object and set the pdf for the fit
unbinned_fit = Fit(data=unbinned_SplusB, model_function=signal_plus_background)

# -- perform the fit
unbinned_fit.do_fit()  

# -- create a plot object
unbinned_plot = Plot(unbinned_fit)  
unbinned_plot.plot(asymmetric_parameter_errors=True)

# compare with binned likelihood fit
SplusB_histogram = HistContainer(n_bins=50, bin_range=(min, max), fill_data=SplusB_data)
hist_fit = Fit(data=SplusB_histogram, model_function=signal_plus_background)
hist_fit.do_fit()  # do the fit
hist_plot = Plot(hist_fit)
hist_plot.plot(asymmetric_parameter_errors=True)

# - show all results
plt.show()
