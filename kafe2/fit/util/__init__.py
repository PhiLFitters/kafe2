r"""This submodule provides utility functions for other modules.

:synopsis: This submodule provides utility functions for other modules.

.. moduleauthor:: Johannes Gaessler <johannes.gaessler@student.kit.edu>
"""

import warnings
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from . import function_library

__all__ = ['string_join_if', 'add_in_quadrature', 'invert_matrix', 'cholesky_decomposition',
           'log_determinant', 'log_determinant_pointwise', 'collect', 'is_diagonal']

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


def log_determinant_pointwise(pointwise_error):
    """
    Calculate the logarithm of the determinant of a matrix from its Cholesky decomposition.
    """
    return 2.0 * np.sum(np.log(pointwise_error))


def collect(*args):
    '''collect arguments into array'''
    return np.asarray(args)


def is_diagonal(matrix):
    """Return True if only diagonal matrix elements are non-zero."""
    return np.all(matrix - np.diag(np.diagonal(matrix)) == 0)

def to_python_floats(yaml_dict: Optional[dict]):
    if yaml_dict is None:
        return None
    _new_dict = {}
    for _key, _value in yaml_dict.items():
        if isinstance(_value, float):
            _value = float(_value)  # Casts NumPy scalars to Python float.
        elif isinstance(_value, np.ndarray):
            _value = _value.tolist()
        elif isinstance(_value, OrderedDict):
            _value = [float(_v) for _v in _value.values()]
        _new_dict[_key] = _value
    return _new_dict

def to_numpy_arrays(yaml_dict: Optional[dict]):
    if yaml_dict is None:
        return None
    _new_dict = {}
    for _key, _value in yaml_dict.items():
        if isinstance(_value, list):
            _value = np.array(_value)
        _new_dict[_key] = _value
    return _new_dict
