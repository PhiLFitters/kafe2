"""Unbinned log-likelihood fit

   fit a Gaussian signal on a flat background

   compare results from binned and unbinned log-likelihood fits
"""


from kafe2 import hist_fit, unbinned_fit, plot
import numpy as np


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
num_bins = 50 # number of bins for the histogram fit
min = 0.0     # range of data, minimum
max = 10.0    # maximum
s = 0.8       # signal fraction
pos = 6.66    # signal position
width = 0.33  # signal width
SplusB_data = generate_data(N, min, max, pos, width, s)  

unbinned_fit(data=SplusB_data, model_function=signal_plus_background)
plot(show=False)  # Set show=False so the plot for the unbinned fit is not shown until the end.

hist_fit(data=SplusB_data, n_bins=50, bin_range=(min, max), model_function=signal_plus_background)
plot()
