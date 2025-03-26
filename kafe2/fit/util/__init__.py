r"""This submodule provides utility functions for other modules.

:synopsis: This submodule provides utility functions for other modules.

.. moduleauthor:: Johannes Gäßler <johannes.gaessler@cern.ch>
"""

# flake8: noqa F401, F403 (imported but unused, used but unable to detect undefined names)

import warnings
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from . import function_library
from .wrapper import *

__all__ = [
    "string_join_if",
    "add_in_quadrature",
    "invert_matrix",
    "cholesky_decomposition",
    "log_determinant",
    "log_determinant_pointwise",
    "collect",
    "is_diagonal",
]

# -- general utility functions


def string_join_if(pieces, delim="_", condition=lambda x: x):
    """Join all elements of `pieces` that pass `condition` together
    using delimiter `delim`."""
    return delim.join((p for p in pieces if condition(p)))


# -- array/matrix utility functions


def add_in_quadrature(*args):
    """return the square root of the sum of squares of all arguments"""
    return np.sqrt(np.sum([_a**2 for _a in args], axis=0))


def invert_matrix(mat):
    """perform matrix inversion"""
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        if mat is not None and np.all(np.isfinite(mat)):
            warnings.warn("Singular covariance matrix. Are the errors for some data points equal to zero?")
        return None


def cholesky_decomposition(mat):
    """
    Perform Cholesky decomposition of a matrix. Covariance matrices != 0 are always symmetric and
    positive-definite.
    """
    try:
        return np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        if mat is not None and np.all(np.isfinite(mat)):
            warnings.warn("Singular covariance matrix. Are the errors for some data points equal to zero?")
        return None


def qr_decomposition(mat):
    """
    Perform QR decomposition of a matrix.
    """
    try:
        return np.linalg.qr(mat)
    except np.linalg.LinAlgError:
        if mat is not None and np.all(np.isfinite(mat)):
            warnings.warn("Singular covariance matrix. Are the errors for some data points equal to zero?")
        return None


def log_determinant_cholesky(cholesky_mat):
    """
    Calculate the logarithm of the determinant of a matrix from its Cholesky decomposition.
    """
    if cholesky_mat is None:
        return 0.0  # Easier to handle for multifits than returning None
    else:
        return 2.0 * np.sum(np.log(np.diag(cholesky_mat)))


def log_determinant_qr(qr):
    """
    Calculate the logarithm of the determinant of a matrix from its QR decomposition.
    """
    if qr is None:
        return 0.0  # Easier to handle for multifits than returning None
    else:
        # det(R) can be negative.
        # det(Q) can be +1 or -1.
        # det(Q @ R) must be positive, so abs(det(R)) == det(Q @ R).
        return np.sum(np.log(np.abs(np.diag(qr[1]))))


def log_determinant_pointwise(pointwise_error):
    """
    Calculate the logarithm of the determinant of a matrix from its Cholesky decomposition.
    """
    return 2.0 * np.sum(np.log(pointwise_error))


def collect(*args):
    """collect arguments into array"""
    return np.asarray(args)


def is_diagonal(matrix):
    """Return True if only diagonal matrix elements are non-zero."""
    return np.all(matrix - np.diag(np.diagonal(matrix)) == 0)


def to_python_types(yaml_dict: Optional[dict]):
    if yaml_dict is None:
        return None
    _new_dict = {}
    for _key, _value in yaml_dict.items():
        if isinstance(_value, float):
            _value = float(_value)  # Casts NumPy scalars to Python float.
        elif isinstance(_value, np.int64):
            _value = int(_value)
        elif isinstance(_value, np.ndarray):
            _value = _value.tolist()
        elif isinstance(_value, OrderedDict):
            _value = [float(_v) if not isinstance(_v, np.ndarray) else _v.tolist() for _v in _value.values()]
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


def check_numerical_range(vals, name):
    if vals is None:
        return
    try:
        _vals = np.asarray(vals)
    except ValueError:
        for _val in vals:
            check_numerical_range(_val, name)
        return
    if _vals.ndim == 0:
        _median_abs = np.abs(_vals)
        if _median_abs == 0:
            return
        _warning_start = "The value of"
    else:
        _filter = np.logical_and(_vals != 0, _vals != None)
        if not np.any(_filter):
            return
        _median_abs = np.median(np.abs(_vals[_filter]))
        _warning_start = "The median absolute non-zero value found in"
    _warning_template = "{start} {name} is very {adjective} with {median_abs:.3e}. Consider whether it's possible to re-scale the fit to avoid any excessively small or large absolute values."
    if _median_abs > 1e9:
        warnings.warn(_warning_template.format(start=_warning_start, name=name, adjective="large", median_abs=_median_abs))
    if _median_abs < 1e-9:
        warnings.warn(_warning_template.format(start=_warning_start, name=name, adjective="small", median_abs=_median_abs))
