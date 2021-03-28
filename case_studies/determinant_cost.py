"""
kafe2 Case Study: Determinant Cost
==================================

So far we've seen two cases where kafe2 uses dynamic errors: when adding x errors or when adding
errors relative to the model.
In such cases the errors are a function of the parameter values.
However, this introduces a bias towards parameter values that result in large errors because this
reduces the overall cost.
More specifically, this results in a bias towards parameter values with increased absolute
derivatives for x errors or with increased absolute model function values for relative model errors.
The aforementioned bias can be counteracted by adding an extra term to the cost function: the
logarithm of the determinant of the covariance matrix.

In this example we will investigate the effect of (not) adding the determinant cost to a chi2 cost
function when handling data with xy errors with kafe2.
To get a better understanding of how kafe2 works internally we will also do a manual implementation
of a fit with SciPy.
Finally, we will compare these results with SciPy orthogonal distance regression (ODR), another tool
that can fit a model function to data with xy errors.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.odr import RealData, Model, ODR
from kafe2 import XYFit, XYCostFunction_Chi2
from numdifftools import Hessian
import matplotlib.pyplot as plt

# Seed the NumPy RNG to ensure consistent results:
np.random.seed(1)

# The x error is much larger than the y error.
# This results in a stronger bias compared to a large y error and a small x error.
# The bias disappears for X_ERROR -> 0.
X_ERROR = 1.0
Y_ERROR = 0.2

# The fit parameter values we use to generate the toy data:
TRUE_PARAMETER_VALUES = np.array([1.0, 0.1, -1.0])
PARAMETER_NAMES = ["a", "b", "c"]


# Our model function is an exponential model with three parameters:
def model_function(x, a, b, c):
    return a * np.exp(b * x) + c


# The derivative of our model function.
# Note that the parameters have different effects on the derivative.
# c has no effect at all.
# An increase in either a or b leads to an increase in the derivative,
# the effect of b is greater for x > 0.
def model_function_derivative(x, a, b, c):
    return a * b * np.exp(x * b)


# The x data assumed by the experimenter:
x_data = np.linspace(start=-10, stop=10, num=61)
# The actual x data when factoring in x errors:
true_x_data = x_data + np.random.normal(size=x_data.shape, scale=X_ERROR)
# The y data based on the unknown true x values:
y_data = model_function(true_x_data, *TRUE_PARAMETER_VALUES)\
         + np.random.normal(size=x_data.shape, scale=Y_ERROR)


# Utility function to do a fit with kafe2:
def kafe2_fit(add_determinant_cost):
    fit = XYFit(
        xy_data=[x_data, y_data],
        model_function=model_function,
        # Create a kafe2 cost function object to turn off the determinant cost:
        cost_function=XYCostFunction_Chi2(add_determinant_cost=add_determinant_cost),
    )
    fit.add_error(axis="x", err_val=X_ERROR)
    fit.add_error(axis="y", err_val=Y_ERROR)
    # Set the parameter values to the true values because we're only interested in the bias:
    fit.set_all_parameter_values(TRUE_PARAMETER_VALUES)
    fit.do_fit()
    return fit.parameter_values, fit.parameter_errors


kafe2_values_det, kafe2_errors_det = kafe2_fit(add_determinant_cost=True)
kafe2_values_no_det, kafe2_errors_no_det = kafe2_fit(add_determinant_cost=False)


# This is our chi2 cost function.
def chi2(args, add_determinant_cost):
    a, b, c = args  # Unpack args from format expected by scipy.optimize.
    y_model = model_function(x_data, a, b, c)

    # Calculate the projected y error by extrapolating the x error based on the derivatives.
    # Note how a large absolute derivative results in a large projected y error.
    projected_y_error = np.sqrt(
        Y_ERROR ** 2
        + (model_function_derivative(x_data, a, b, c) * X_ERROR) ** 2
    )

    # Now just calculate chi2 as per usual:
    normed_residuals = (y_data - y_model) / projected_y_error
    cost = np.sum(normed_residuals ** 2)
    # Note how large values for projected_y_error result in a lower overall cost.

    if add_determinant_cost:
        # Add extra cost based on the determinant of the covariance matrix.
        # We are using uncorrelated errors in which case the covariance matrix is diagonal.
        # The determinant can therefore be calculated as np.prod(projected_y_error ** 2) .
        # But because this can result in overflow we instead calculate the extra cost like this:
        cost += 2.0 * np.sum(np.log(projected_y_error))
        # The above line is equivalent to:
        # cost += np.log(np.prod(projected_y_error ** 2))
        #
        # Note how large values for projected_y_error result in a higher overall cost.
    return cost


# Utility function to do a manual fit with scipy.optimize.minimize:
def scipy_fit(add_determinant_cost):
    # Wrapped function to hand over to minimize:
    def cost_function(args):
        return chi2(args, add_determinant_cost)

    optimize_result = minimize(
        fun=cost_function,
        # Initialize fit with true values because we're only interested in the bias:
        x0=TRUE_PARAMETER_VALUES,
    )
    parameter_values = optimize_result.x

    # Calculate parameter errors from Hessian matrix (2nd order derivatives) at minimum:
    hessian_matrix = Hessian(cost_function)(parameter_values)
    # Not to be confused with the covariance matrix of our data:
    parameter_covariance_matrix = 2.0 * np.linalg.inv(hessian_matrix)
    parameter_errors = np.sqrt(np.diag(parameter_covariance_matrix))
    return parameter_values, parameter_errors


scipy_values_det, scipy_errors_det = scipy_fit(add_determinant_cost=True)
scipy_values_no_det, scipy_errors_no_det = scipy_fit(add_determinant_cost=False)

# Do a fit with SciPy ODR for comparison:
odr_data = RealData(x_data, y_data, X_ERROR, Y_ERROR)
odr_model = Model(lambda parameter_values, x: model_function(x, *parameter_values))
odr_fit = ODR(odr_data, odr_model, beta0=TRUE_PARAMETER_VALUES)
odr_result = odr_fit.run()
odr_values = odr_result.beta
odr_errors = odr_result.sd_beta


# Utility function to print out results:
def print_results(name, parameter_values, parameter_errors):
    print("======== {name} ========".format(name=name))
    for pn, pv, pe, epv in zip(
            PARAMETER_NAMES, parameter_values, parameter_errors, TRUE_PARAMETER_VALUES):
        sigma = abs(pv - epv) / pe
        print("{pn} = {pv:.4f} +- {pe:.4f} (off by {sigma:.2f} sigma)".format(
            pn=pn, pv=pv, pe=pe, sigma=sigma))
    print()


print_results("kafe2 with det cost", kafe2_values_det, kafe2_errors_det)
print_results("kafe2 without det cost", kafe2_values_no_det, kafe2_errors_no_det)
print_results("scipy minimize with det cost", scipy_values_det, scipy_errors_det)
print_results("scipy minimize without det cost", scipy_values_no_det, scipy_errors_no_det)
# Unsurprisingly kafe2 and our manual re-implementation of kafe2 yield almost the exact same result.
# Note how fits without determinant cost have higher values for b which has the biggest influence
# on the model function derivative.
# Because our fit parameters are correlated this bias also influences the other parameters,
# even if they have no influence on the model function derivative as is the case with c.
#
# With the default seed our fit result becomes worse through the aforementioned bias.
# However, if the seed is changed or removed the biased fit result can sometimes be better.
# This is because our data can just happen to have errors that result in a fit with parameter values
# that underestimate the model function derivative.
# On average the unbiased fit result will be better.

print_results("scipy ODR", odr_values, odr_errors)
# SciPy ODR is comparable to kafe2 without determinant cost.

# Finally, let's do a simple plot for our results.
# Note how the fit result without determinant cost results in a steeper model function.
x_plot = np.linspace(start=-12, stop=12, num=121)
plt.errorbar(
    x_data, y_data, xerr=X_ERROR, yerr=Y_ERROR,
    color="tab:blue", marker=".", ls="", label="Data"
)
plt.plot(
    x_plot, model_function(x_plot, *kafe2_values_det),
    color="black", label="kafe2 with det cost"
)
plt.plot(
    x_plot, model_function(x_plot, *kafe2_values_no_det),
    color="yellow", ls="--", label="kafe2 without det cost"
)
plt.plot(
    x_plot, model_function(x_plot, *odr_values),
    color="red", ls=":", label="scipy odr"
)
plt.legend()
plt.xlim(-12, 12)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
