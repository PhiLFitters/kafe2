import numpy as np
#TODO documentation

__all__ = ['linear_model']

def linear_model(x, a, b):
    return a * x + b

def normal_distribution_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)
    