import numpy as np
#TODO documentation

__all__ = ['linear_model']

def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x ** 2 + b * x + c

def cubic_model(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def exponential_model(x, A0, x0):
    return A0 * np.exp(x / x0)

def normal_distribution_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)
    
STRING_TO_FUNCTION = {
    'linear':linear_model,
    'linear_model':linear_model,
    'quadratic':quadratic_model,
    'quadratic_model':quadratic_model,
    'cubic':cubic_model,
    'cubic_model':cubic_model,
    'exp':exponential_model,
    'exponential':exponential_model,
    'exponential_model':exponential_model,
    'normal':normal_distribution_pdf,
    'normal_distribution_pdf':normal_distribution_pdf,
}
