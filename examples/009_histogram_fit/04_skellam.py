#!/usr/bin/env python
"""
kafe2 kellam Histogram Fit: signal on flat background
=====================================================

Sometimes, when a signal over background fit has to be performed, it is hard to actually
parametrize the background. A promnent example is e.g. gamma spectoscropy, where we
could want to perform a fit on the photopeak.
In these cases we often record a background spectrum often over a longer period of time
with no source present. This background is then rescaled to the recording time of the 
signal + background measurement and subtracted.

Both the signal + background measurement as well as the background measurments will
then follow a poisson distribution in each bin. This poisson nature is lost as soon
as we do the subtraction which is why we have to then use a so called skellam distibubtion
to calculate our likelihood.

The following example showcases, how to perform such a fit and compares the results to a
combined signal + background fit. For this reason, the background specttrum is just chosen
to be a simple flat background, but in principle also more complex backgrounds can be 
handled with this method.     
"""

import numpy as np
import matplotlib.pyplot as plt
from kafe2 import HistContainer, Fit, Plot, HistFit

N = 200  # number of entries
min = 0.0  # range of data, minimum
max = 10.0  # maximum
s = 0.8  # signal fraction
pos = 6.66  # signal position
width = 0.33  # signal width
beg_sig = 5.5
end_sig = 8
background_est_frac = 100

def generate_data(N=100, min=0, max=1.0, pos=0.0, width=0.25, signal_fraction=0.1, background_est_frac=1.0):
    """generate a random dataset:
    Gaussian signal at position p with width w and signal fraction s
    on top of a flat background between min and max
    and a recorded background spectrum that can be scaled by backgorund_est_frac
    """
    # signal sample
    data_s = np.random.normal(loc=pos, scale=width, size=int(signal_fraction * N))
    # background sample
    data_b = np.random.uniform(low=min, high=max, size=int((1 - signal_fraction) * N))
    #isolated background measurement
    data_b_est = np.random.uniform(low=min, high=max, size=int((1 - signal_fraction) * N * background_est_frac) )
    return np.concatenate((data_s, data_b)), data_b_est

def s_plus_b(x, Ns = 200, mu=3.0, sigma=2.0, Nb = 50.):
    """Gaussian signal on top of flat background"""
    normal = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma**2)
    flat = 1.0 / (max - min)
    return Ns * normal + Nb * flat

def signal(x, Ns = 160, mu=6, sigma = 0.3):
   """Gaussian signal"""
   normal = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma**2)
   return Ns * normal
   


if __name__=="__main__": # -----------------------------


  
  # generate a histogram data sample, for the combined fit we won't need the backgorund estimation data
  SplusB_data, background_est= generate_data(N, min, max, pos, width, s, background_est_frac=background_est_frac)

  #Create a histogram Container from the dataset
  SplusB_histcontainer = HistContainer(n_bins=35, bin_range=(min, max), fill_data=SplusB_data)
  SplusB_histcontainer.label = "artifical data"

  # First perform the combined fit
  hist_fit = Fit(data=SplusB_histcontainer, density=False,
                  model_function=s_plus_b)
  
  hist_fit.do_fit()  # 1st fit
  hist_fit.report()  

  # Optional: create plot and show it
  hist_plot = Plot([ hist_fit])
  hist_plot.plot(asymmetric_parameter_errors=True)
  plt.show()

  #Now we perform the skellam fit 
  #First we select a signal region in which we want to fit the signal distribution

  signal_region_data = SplusB_data[(SplusB_data > beg_sig) & (SplusB_data < end_sig)]
  signal_region_background = background_est[(background_est > beg_sig) & (background_est < end_sig)]

  # Fill the data into histograms and subtract the rescaled background
  SplusB_binned, bins = np.histogram(signal_region_data, bins = 8, range = (beg_sig,end_sig))
  background_binned, bins = np.histogram(signal_region_background, bins = 8, range = (beg_sig,end_sig))
  background_binned_rescaled = background_binned * 1/background_est_frac
  signal_binned = SplusB_binned - background_binned_rescaled
  #signal_binned[signal_binned <0] = 0

  #fill data into container and define fit object with background estimation and correct cost function
  signal_container = HistContainer(bin_edges = bins)
  signal_container.set_bins(signal_binned)
  signal_fit = HistFit(signal_container, model_function=signal, cost_function='skellam', density=False, background_array=background_binned_rescaled)

  signal_fit.do_fit() #2nd fit
  signal_fit.report()

  plot = Plot(signal_fit)
  plot.plot()
  plt.show()



