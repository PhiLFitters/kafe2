"""Perform a simultaneous fit to two frequency distributions
   (= histograms) with common parameters with kafe2.MultiFit()

   This example illustrates another common use-case for multifits,
   where the same signal is measured under varying conditions,
   e.g. in different detector regions with different resolutions
   and background levels.

   Consider the distribution of a signal on top of a flat background.
   Additional smearing is added to the "true" data values. A second,
   similar set of data at the same position and with the same width
   is generated, albeit with a differing number of signal events,
   smaller signal fraction and less resolution smearing.

   A simultaneous fit using the kafe2 MultiFit feature is then performed
   to extract the position and raw width common to the two data sets.

   *Note*: in this simple case of two independent frequency distributions
   the results for the common parameters could also be determined by
   combination of the results from two individual fits to each of the
   histograms.
"""

from kafe2 import Fit, Plot, HistContainer, MultiFit
import numpy as np


# function fo generate the signal-plus-background distributions
def generate_data(N, min, max, pos, width, s):
    """generate a random dataset:
     Gaussian signal at position p with width w and signal fraction s
     on top of a flat background between min and max
    """
    # signal sample
    data_s = np.random.normal(loc=pos, scale=width, size=int(s * N))
    # background sample
    data_b = np.random.uniform(low=min, high=max, size=int((1 - s) * N))
    return np.concatenate((data_s, data_b))


# the fit functions, one for each version of the distribution with
#  different resolution and signal fraction
#
def SplusBmodel1(x, mu=5., width=0.3, res1=0.3, sf1=0.5):
    """pdf of a Gaussian signal at position mu, with natural width width,
    resolution res1 and signal fraction sf1 on a flat background
    """
    sigma2 = width * width + res1 * res1
    normal = np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2.0 * np.pi * sigma2)
    flat = 1. / (max - min)
    return sf1 * normal + (1 - sf1) * flat


def SplusBmodel2(x, mu=5., width=0.3, res2=0.3, sf2=0.5):
    """pdf of a Gaussian signal at position mu, with natural width width,
    resolution res2 and signal fraction sf2 on a flat background
    """
    sigma2 = width * width + res2 * res2
    normal = np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2.0 * np.pi * sigma2)
    flat = 1. / (max - min)
    return sf2 * normal + (1 - sf2) * flat


# --- generate data sets, set up and perform fit
min = 0.
max = 10.
pos = 6.66
width = 0.33

# -- generate a first data set 
s1 = 0.8
N1 = 200
r1 = 2 * width  # smearing twice as large as natural width
SplusB_raw1 = generate_data(N1, min, max, pos, width, s1)
# apply resolution smearing to data set SplusB_data 
SplusB_data1 = SplusB_raw1 + np.random.normal(loc=0., scale=r1, size=len(SplusB_raw1))

# -- generate a second data set at the same position and width, 
#   but with smaller signal fraction, better resolution and more events 
s2 = 0.25
N2 = 500
r2 = width / 3.
SplusB_raw2 = generate_data(N2, min, max, pos, width, s2)
SplusB_data2 = SplusB_raw2 + np.random.normal(loc=0., scale=r2, size=len(SplusB_raw2))

# -- Create histogram containers from the two datasets
SplusB_histogram1 = HistContainer(n_bins=30, bin_range=(min, max), fill_data=SplusB_data1)
SplusB_histogram2 = HistContainer(n_bins=50, bin_range=(min, max), fill_data=SplusB_data2)

# -- create Fit objects by specifying their density functions with corresponding parameters
hist_fit1 = Fit(data=SplusB_histogram1, model_function=SplusBmodel1)
hist_fit2 = Fit(data=SplusB_histogram2, model_function=SplusBmodel2)
# to make the fit unambiguous,
#  external knowledge on the resolutions must be applied
hist_fit1.add_parameter_constraint(name='res1', value=r1, uncertainty=r1 / 4.)
hist_fit2.add_parameter_constraint(name='res2', value=r2, uncertainty=r2 / 2.)

# -- test: perform individual fits
print('\n*==* Result of fit to first histogram')
hist_fit1.do_fit()
hist_fit1.report()
print('\n*==* Result of fit to second histogram')
hist_fit2.do_fit()
hist_fit2.report()

# combine the two fits to a MultiFit
multi_fit = MultiFit(fit_list=[hist_fit1, hist_fit2])

multi_fit.do_fit()  # do the fit
print('\n*==*  Result of multi-fit to both histograms')
multi_fit.report()  # Optional: print a report to the terminal

# Optional: create output graphics
multi_plot = Plot(multi_fit, separate_figures=True)
multi_plot.plot(asymmetric_parameter_errors=True)

multi_plot.show()
