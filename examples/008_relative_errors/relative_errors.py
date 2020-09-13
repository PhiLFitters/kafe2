#!/usr/bin/env python
"""
kafe2 example: Relative errors
==============================

Like many tools kafe2 supports specifying errors that are relative to measurement data: the errors
are calculated as the simple product of the absolute values of the data and the error values.
However, this approach results in a bias. Measurements whose absolute value is reduced from the
measurement error are assigned a smaller uncertainty than the measurements whose absolute value is
increased by the measurement error. As a consequence some measurements are incorrectly given a
higher weight than other measurements.

To fix the above issue kafe2 allows users to specify errors relative to the model instead of the
data: the errors are then calculated dynamically as the product of the absolute values of the model
and the error values. The model is our hypothesis for the "true values" and is therefore unaffected
by the difference in measurement error between two data points.

In this example we are performing a very simple analysis: we simulate some amount of measurements
for a parameter with some relative Gaussian measurement errors. We then simply average the results
and compare the estimate for the parameter to the "true value". We will find that with the default
settings of this example specifying relative errors can introduce a bias that is much larger than
the estimated uncertainty for the average.

If you want to know the details:
To improve convergence kafe2 iteratively performs at least two fits when errors relative to model
are specified. The first fit is performed with static errors relative to data.
This fixes some problems related to a bad choice of
starting parameters. In particular it fixes the inability to cross a model value of 0 due to the
cost function value exploding from the relative errors approaching 0.
After the first fit more fits with errors relative to model are performed to correct the bias of the
first fit. If dynamic_error_algorithm is set to "nonlinear" only one extra fit is performed where
the errors are updated during the minimization. If dynamic_error_algorithm is instead set to
"iterative" multiple fits with static relative model errors are performed iteratively until the cost
function value converges (relative model errors are updated between fits).
"""

import numpy as np
from kafe2 import IndexedFit

data_size = 1000  # The number of measurements to average.
true_mean = 1.0  # The mean value of the data.
initial_par_value = -0.5  # The initial guess for the average. Note the different sign.
relative_error = 0.1  # The relative error on the data.

np.random.seed(10)
# Simulate data with Gaussian measurement errors:
data = true_mean * (np.ones(data_size) + relative_error * np.random.randn(data_size))


# Simple model function to average our measurement values:
def model_function(a=initial_par_value):
    return np.ones(data_size) * a


# Utility function to create a fit
def get_fit(reference, dynamic_error_algorithm="nonlinear"):
    fit = IndexedFit(data=data, model_function=model_function,
                     dynamic_error_algorithm=dynamic_error_algorithm)
    # Model errors MUST be specified using a fit object, not a data container object:
    fit.add_error(relative_error, relative=True, reference=reference)
    return fit


# Fit with uncertainties relative to data:
fit_data_err = get_fit(reference="data")
results_data_err = fit_data_err.do_fit()

# Fit with uncertainties relative to model (nonlinear):
fit_model_err_nonlinear = get_fit(reference="model", dynamic_error_algorithm="nonlinear")
results_model_err_nonlinear = fit_model_err_nonlinear.do_fit()

# Fit with uncertainties relative to model (iterative):
fit_model_err_iterative = get_fit(reference="model", dynamic_error_algorithm="iterative")
results_model_err_iterative = fit_model_err_iterative.do_fit()


# Utility function to print out the relevant parts of the fit results:
def print_results(results, name):
    print("========== Error relative to %s ==========" % name)
    mean = results["parameter_values"]["a"]
    std = results["parameter_errors"]["a"]
    print("Fit result: %.4f +- %.4f" % (mean, std))
    print("Difference between fit result and true mean: %.2f%% (%.2f sigma)" % (
        100 * abs((true_mean - mean) / true_mean),
        abs((true_mean - mean) / std)
    ))
    print()


# The fit with relative data errors is biased towards a smaller average:
print_results(results_data_err, name="data")
# The fit with relative model errors (nonlinear) is less biased towards a larger average:
print_results(results_model_err_nonlinear, name="model (nonlinear)")
# The fit with relative model errors (iterative) is essentially unbiased but it's incompatible
# with asymmetric parameter errors, parameter profiles, and parameter contours:
print_results(results_model_err_iterative, name="model (iterative)")

