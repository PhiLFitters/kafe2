import numpy as np

from kafe2 import XYFitEnsemble
from kafe2.tools import print_dict_recursive

import matplotlib.pyplot as plt

# def xy_model(x, a, b, c):
#     return a * x ** 2 + b * x + (c**3)*0.01
# reference_model_parameters = 1., -3., 42.

def xy_model(x, a, b):
    return a * x + b

reference_model_parameters = 1., 42.

# def xy_model(x, a):
#     return a * x
# reference_model_parameters = 1.,

reference_x_support_points = np.arange(1, 7)

RESULTS_TO_COLLECT = ['y_data', 'y_pulls', 'parameter_pulls']

fit_ensemble = XYFitEnsemble(n_experiments=1000,
                             x_support=reference_x_support_points,
                             model_function=xy_model,
                             model_parameters=reference_model_parameters,
                             requested_results=RESULTS_TO_COLLECT)

# -- specify error model: we will test two scenarios

# Scenario 1: use the measurement data as a reference
#             for converting *relative* errors into
#             the *absolute* errors used in the fit

RELATIVE_TO_WHAT = 'data'

# Scenario 2: use the *fitted model* as a reference
#             for converting *relative* errors into
#             *absolute* errors.

# RELATIVE_TO_WHAT = 'model'

# -- add a single, uncorrelated, relative *y* error of 30%

fit_ensemble.add_simple_error(
    axis='y',
    err_val=0.3,
    correlation=0.0,
    relative=True,
    reference=RELATIVE_TO_WHAT,
)

# -- do the fits for the ensemble
fit_ensemble.run()

fit_ensemble._toy_fit.report()

# -- calculate some statistics of the collected ensemble variables:
#    sample mean, sample standard deviation and standard error of the mean
_stats = fit_ensemble.get_results_statistics(statistics=['mean', 'std', 'mean_error'])

# -- print out the statistics
print("Statistics")
print_dict_recursive(_stats)

# -- plot histograms of the marginal distributions of the ensemble variables
fit_ensemble.plot_result_distributions(
    results=RESULTS_TO_COLLECT
)

# -- make scatter plots of the collected ensemble variables
fit_ensemble.plot_result_scatter(
    results=RESULTS_TO_COLLECT
)

# -- show the plots
plt.show()