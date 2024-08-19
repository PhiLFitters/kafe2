#!/usr/bin/env python
"""
kafe2 example: Model Functions
==============================

In experimental physics a line fit will only suffice for a small number
of applications. In most cases you will need a more complex model function
with more parameters to accurately model physical reality.
This example demonstrates how to specify arbitrary model functions for
a kafe2 fit.
"""

import numpy as np
import kafe2


# To define a model function for kafe2 simply write it as a Python function.
# Important: The first argument of the model function is interpreted as the independent variable
#     of the fit. It is not being modified during the fit and it's the quantity represented by
#     the x axis of our fit.

# Our first model is a simple linear function:
def linear_model(x, a, b):
    return a * x + b


# If SymPy is installed you can also define the model like this:
# linear_model = "linear_model: x a b -> a * x + b"


# Our second model is a simple exponential function.
# The kwargs in the function header specify parameter defaults.
def exponential_model(x, A_0=1., x_0=5.):
    return A_0 * np.exp(x/x_0)


# If SymPy is installed you can also define the model like this:
# exponential_model = "exponential_model: x A_0 x_0=5.0 -> A_0 * exp(x / x_0)"


x_data = [0.38, 0.83, 1.96, 2.82, 4.28, 4.69, 5.97, 7.60, 7.62, 8.81, 9.87, 10.91]
y_data = [1.60, 1.66, 2.12, 3.05, 3.57, 4.65, 6.21, 7.57, 8.27, 10.79, 14.27, 18.48]
x_error = 0.2  # 0.2 in absolute units of x
y_error = 0.5  # 0.5 in absolute units of y
y_error_rel = 0.03  # 3% of the y model value

# kafe2.xy_fit needs to be called twice to do two fits:
kafe2.xy_fit(linear_model, x_data, y_data,
             x_error=x_error, y_error=y_error, y_error_rel=y_error_rel)
kafe2.xy_fit(exponential_model, x_data, y_data,
             x_error=x_error, y_error=y_error, y_error_rel=y_error_rel)
# Make sure to specify profile=True whenever you use a nonlinear model function.
# A model function is linear if it is a linear function of each of its parameters.
# The model function does not need to be a linear function of the independent variable x.
# Examples: all polynomial model functions are linear, trigonometric functions are nonlinear.

# To specify that you want a plot of the last two fits pass -2 as the first argument:
kafe2.plot(
    -2,

    # Uncomment the following line to use different names for the parameters:
    # parameter_names=dict(x="t", a=r"\alpha", b=r"\beta", A_0="I_0", x_0="t_0"),
    # Use LaTeX for special characters like greek letters.

    # When Python functions are used as custom model functions kafe2 does not know
    # how to express them as LaTeX. The LaTeX can be manually defined like this:
    model_expression=["{a}{x} + {b}", "{A_0} e^{{{x}/{x_0}}}"],
    # Parameter names have to be put between {}. To get {} for LaTex double them like {{ or }}.
    # When using SymPy to define model function the LaTeX expression can be derived automatically.

    # Add a so-called pull plot in order to compare the influence of individual data points on the models.
    extra="pull"
    # Alternative extra plots: "residual" or "ratio".
)
