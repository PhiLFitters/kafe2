#!/usr/bin/env python2
# -*- coding: utf8 -*-

# some things we need
import numpy as np
import scipy.stats as stats

# import main stuff from kafe
from kafe.fit import (
    IndexedContainer, IndexedFit, IndexedPlot,
    XYContainer, XYFit, XYPlot,
    HistContainer, HistFit, HistPlot
)

# important: import matplotlib *after* kafe!
import matplotlib.pyplot as plt
from kafe.fit._base.profile import ContoursProfiler
from kafe.fit.xy.cost import XYCostFunction_Chi2


# -- models

# a simple quadratic 'xy' model (3 parameters)
def xy_model(x, a, b, c):
    return a * x ** 2 + b * x + c

# a simple 'indexed' model: i-th value is xy_model(i, a, b, c)
def idx_model(a, b, c):
    return xy_model(np.arange(10), a, b, c)

# histogram model: probability density
def norm_pdf(x, mu=0.0, sigma=1.0):
    return stats.norm(mu, sigma).pdf(x)

# histogram model: cumulative distribution function
def norm_cdf(x, mu=0.0, sigma=1.0):
    return stats.norm(mu, sigma).cdf(x)



# -- cost functions

# a simple 'chi-square' cost function
def chi2(data, model):
    return np.sum((data - model) ** 2)

# a simple 'chi-square' cost function with 'xy' nomenclature
def chi2_xy(y_data, y_model):
    return np.sum((y_data - y_model) ** 2)


def example_indexed_fit():
    """Workflow for a kafe fit to an 'Indexed' data set"""
    # -- prerequisites

    # set the quadratic model parameters to numeric values
    a0, b0, c0 = 0.1, 0.2, 3.3

    # compute pseudo-data for 'indexed' model: model prediction + random gaussian noise
    data0 = idx_model(a0, b0, c0) + np.random.normal(0, 1, 10)


    # -- do some kafe fits: IndexedFit

    fits = []  # store our kafe 'Fit' objects here
    for i in xrange(4):
        # shift data in 'y' direction, so the different fits do not overlap
        _data_with_offset = data0 + i

        # initialize an 'IndexedFit'
        f = IndexedFit(data=_data_with_offset,
                       model_function=idx_model,
                       cost_function=chi2)

        # give parameters (trivial) LaTeX names
        f.assign_parameter_latex_names(a='a', b='b', c='c')

        # !! doesn't work yet !!
        # f.assign_model_function_expression("{0}*{i}^2 + {1}*{i} + {2}")
        # f.assign_model_function_latex_expression(r"{0}\,x_{i}^2 + {1}\,x_{i} + {2}")

        # add an error source to the 'Fit' object error model
        f.add_simple_error(1.0)  # all points have a (Gaussian) uncertainty of +/-1.0

        # do the fit
        f.do_fit()

        # store the result
        fits.append(f)

    # -- do some plotting

    # create a 'Plot' object: takes a 'Fit' or list of 'Fits' as an argument
    p = IndexedPlot(fit_objects=fits)

    # do the plotting
    p.plot()

    # show fit info in the plot
    p.show_fit_info_box(format_as_latex=True)


def example_histogram_fit():
    """Workflow for a kafe fit to a 'Histogram' data set"""
    # -- prerequisites

    # set the quadratic model parameters to numeric values
    mu0, sigma0 = 1.2, 0.35
    n_total = 1000  # total number of entries in histogram

    # raw data for histogram -> 1000 random points with a Gaussian distribution
    h_data = np.random.normal(mu0, sigma0, n_total)

    # make a histogram from the raw data
    hist_cont = HistContainer(10, (0, 3), fill_data=h_data)

    # initialize a 'HistFit' for fitting a histogram:
    f = HistFit(data=hist_cont,
                model_density_function=norm_pdf,
                cost_function=chi2,
                model_density_antiderivative=norm_cdf)

    # assign LaTeX names to the fit parameters
    f.assign_parameter_latex_names(mu=r'\mu', sigma='\sigma')

    # assign plain-text and LaTeX expressions to the model function
    f.assign_model_function_expression("norm.pdf({0}, {1})({x})")
    f.assign_model_function_latex_expression(r"\mathcal{{N}}\left({0}, {1}\right)\left({x}\right)")

    # add a gaussian error to the fit
    f.do_fit()


    # plot the 'HistFit' with 'HistPlot'
    p = HistPlot(fit_objects=f)
    p.plot()
    p.show_fit_info_box(format_as_latex=True)



def example_xy_fit():
    """Workflow for a kafe fit to an 'XY' data set"""

    # set the quadratic model parameters to numeric values
    a0, b0, c0 = 0.1, 0.2, 3.3

    # compute pseudo-data for 'xy' model: same as 'Indexed', but 'x' is part of data!
    idx_data0 = idx_model(a0, b0, c0) + np.random.normal(0, 1, 10)
    xydata0 = np.array([np.arange(10), idx_data0])


    # -- do some kafe fits: XYFit

    fits = []  # store our kafe 'Fit' objects here
    for i in xrange(4):
        # shift 'y' data in 'y' direction, so the different fits do not overlap
        _y_data_with_offset = xydata0[1] + i
        _xydata_with_offset = np.array([xydata0[0], _y_data_with_offset])

        # initialize an 'IndexedFit'
        f = XYFit(xy_data=_xydata_with_offset,
                  model_function=xy_model)

        # give parameters (trivial) LaTeX names
        f.assign_parameter_latex_names(a='a', b='b', c='c')

        # assign strings for the function expression
        f.assign_model_function_expression("{0}*{x}^2 + {1}*{x} + {2}")
        f.assign_model_function_latex_expression(r"{0}\,{x}^2 + {1}\,{x} + {2}")

        # add an error source to the 'Fit' object error model
        f.add_simple_error('y', 0.25, correlation=0.01)  # all points have a (Gaussian) uncertainty in 'y' of +/-1.0
        f.add_simple_error('x', 0.25, correlation=0.01)

        # do the fit
        f.do_fit()

        # store the result
        fits.append(f)


    # create the plots
    p = XYPlot(fit_objects=fits)
    p.plot()
    p.show_fit_info_box(format_as_latex=True)


def exponential(x, A0=1, tau=1):
    return A0 * np.exp(-x / tau)

def test_implementation():
    a0, b0, c0 = 1.0, 1.0, 1.0
    _x = np.asarray([8.018943e-01, 1.839664e+00, 1.941974e+00, 1.953903e+00, 2.839654e+00, 3.488302e+00, 3.775855e+00, 4.555187e+00, 4.477186e+00, 5.376026e+00])
    _y = np.asarray([2.650644e-01, 1.472682e-01, 8.077234e-02, 1.850181e-01, 5.326301e-02, 1.984233e-02, 1.866309e-02, 1.230001e-02, 9.694612e-03, 2.412357e-03])

    f = XYFit(xy_data=[_x, _y], model_function=exponential, cost_function=XYCostFunction_Chi2(errors_to_use='covariance', axes_to_use='xy', fallback_on_singular=False))
    f.add_simple_error('x', 0.3, correlation=0)
    f.add_simple_error('y', 0.4 * _y, correlation=0)    
    f.do_fit()
    
    print f.parameter_values
    print f.parameter_errors
    print f.cost_function_value
   
    cpf = ContoursProfiler(f)
    cpf.plot_profiles_contours_matrix(show_ticks_for="all")
     
    p = XYPlot(fit_objects=f)
    p.plot()
    p.show_fit_info_box(format_as_latex=True)
    
if __name__ == "__main__":
    # run example workflows
#     example_indexed_fit()
#     example_xy_fit()
#     example_histogram_fit()
    
    # show results
    test_implementation()
    plt.show()
