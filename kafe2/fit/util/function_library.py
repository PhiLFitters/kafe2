import numpy as np
#TODO documentation

__all__ = ['linear_model', 'linear_model_derivative', 'quadratic_model', 'quadratic_model_derivative',
           'cubic_model', 'cubic_model_derivative', 'exponential_model', 'normal_distribution']


def linear_model(x, a, b):
    return a * x + b


def linear_model_derivative(x, a, b):
    return np.ones_like(x) * a


linear_model.expression_format_string = "{a} * {x} + {b}"
linear_model.latex_expression_format_string = r"{a} \, {x} + {b}"


def quadratic_model(x, a, b, c):
    return a * x ** 2 + b * x + c


def quadratic_model_derivative(x, a, b, c):
    return 2 * a * x + b


quadratic_model.expression_format_string = "{a} * {x} ** 2 + {b} * {x} + {c}"
quadratic_model.latex_expression_format_string = r"{a} \, {x}^2 + {b} \, {x} + {c}"


def cubic_model(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def cubic_model_derivative(x, a, b, c, d):
    return 3 * a * x ** 2 + 2 * b * x + c


cubic_model.expression_format_string = "{a} * {x} ** 3 + {b} * {x} ** 2 + {c} * {x} + {d}"
cubic_model.latex_expression_format_string = r"{a} \, {x}^3 + {b} \, {x}^2 + {c} \, {x} + {d}"


def exponential_model(x, A_0, x_0):
    return A_0 * np.exp(x / x_0)


def exponential_model_derivative(x, A_0, x_0):
    return A_0 * np.exp(x / x_0)


exponential_model.expression_format_string = "{A_0} * exp({x} / {x_0})"
exponential_model.latex_expression_format_string = r"{A_0} \, e^{{{x} \ / {x_0}}}"


def normal_distribution(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)


normal_distribution.expression_format_string = \
    "1 / sqrt(2 * pi * {sigma}) * exp(-0.5 * (({x} - {mu}) / {sigma}) ** 2)"
normal_distribution.latex_expression_format_string = \
    (r"\frac{{1}}{{2 \pi {sigma}}} e^{{- \frac{{1}}{{2}}"
    r"\left( \frac{{{x} - {mu}}}{{{sigma}}} \right)^2 }}")

normal_distribution_pdf = normal_distribution  # Backwards compatibility


STRING_TO_FUNCTION = {
    'linear': linear_model,
    'linear_model': linear_model,
    'quadratic': quadratic_model,
    'quadratic_model': quadratic_model,
    'cubic': cubic_model,
    'cubic_model': cubic_model,
    'exp': exponential_model,
    'exponential': exponential_model,
    'exponential_model': exponential_model,
    'normal': normal_distribution,
    'normal_distribution': normal_distribution,
    'normal_distribution_pdf': normal_distribution,
}
