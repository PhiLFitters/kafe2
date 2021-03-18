r"""This submodule provides utility functions for other modules.

:synopsis: This submodule provides utility functions for other modules.

.. moduleauthor:: Johannes Gaessler <johannes.gaessler@student.kit.edu>
"""

import warnings
import numpy as np

from . import function_library

__all__ = ['string_join_if', 'add_in_quadrature', 'invert_matrix', 'cholesky_decomposition', 'log_determinant',
           'collect']

# -- general utility functions

def string_join_if(pieces, delim='_', condition=lambda x: x):
    '''Join all elements of `pieces` that pass `condition` together
    using delimiter `delim`.'''
    return delim.join((p for p in pieces if condition(p)))


# -- array/matrix utility functions

def add_in_quadrature(*args):
    '''return the square root of the sum of squares of all arguments'''
    return np.sqrt(np.sum([_a**2 for _a in args], axis=0))


def invert_matrix(mat):
    '''perform matrix inversion'''
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Singular covariance matrix. Are the errors for some data points equal to zero?")
        return None


def cholesky_decomposition(mat):
    """
    Perform Cholesky decomposition of a matrix. Covariance matrices != 0 are always symmetric and
    positive-definite.
    """
    try:
        return np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Singular covariance matrix. Are the errors for some data points equal to zero?")
        return None


def log_determinant(cholesky_mat):
    """
    Calculate the logarithm of the determinant of a matrix from its Cholesky decomposition.
    """
    if cholesky_mat is None:
        return 0.0  # Easier to handle for multifits than returning None
    else:
        return 2.0 * np.sum(np.log(np.diag(cholesky_mat)))


def collect(*args):
    '''collect arguments into array'''
    return np.asarray(args)
