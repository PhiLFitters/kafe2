import numpy as np
#TODO documentation

__all__ = ['linear_model', 'linear_model_derivative', 'quadratic_model', 'quadratic_model_derivative',
           'cubic_model', 'cubic_model_derivative', 'exponential_model', 'normal_distribution_pdf']


def linear_model(x, a, b):
    return a * x + b


def linear_model_derivative(x, a, b):
    return np.ones_like(x) * a


def quadratic_model(x, a, b, c):
    return a * x ** 2 + b * x + c


def quadratic_model_derivative(x, a, b, c):
    return 2 * a * x + b


def cubic_model(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def cubic_model_derivative(x, a, b, c, d):
    return 3 * a * x ** 2 + 2 * b * x + c


def exponential_model(x, A0, x0):
    return A0 * np.exp(x / x0)


def exponential_model_derivative(x, A0, x0):
    return A0 * np.exp(x / x0)


def normal_distribution_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)


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
    'normal': normal_distribution_pdf,
    'normal_distribution_pdf': normal_distribution_pdf,
}
