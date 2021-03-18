"""
Classes for handling of uncertainties in kafe2 fits.
"""

import abc
import numpy as np
import six
import warnings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig()

__all__ = ['CovMat', 'SimpleGaussianError', 'MatrixGaussianError']


def cov_mat_from_float(value, size, correlation=0.0):
    _val = float(value)
    return cov_mat_from_float_list([_val] * size, correlation)


def cov_mat_from_float_list(value_list, correlation=0.0):
    _vals = np.array(list(map(float, value_list)))
    correlation = float(correlation)
    if not (0.0 <= correlation <= 1.0):
        raise ValueError("Correlation must be between 0 and 1: %g given." % (correlation,))
    if correlation == 1.0:
        _mat = np.outer(_vals, _vals)
    elif correlation == 0.0:
        _mat = np.diag(_vals**2)
    else:
        _mat = np.diag(_vals**2 * (1.0 - correlation))
        _mat += np.outer(_vals, _vals) * correlation
    return CovMat(_mat)


# Data structure for Covariance Matrices
class CovMat(object):
    def __init__(self, matrix):
        self.mat = matrix

    # -- 'magic' methods

    def __iadd__(self, other):
        self.mat += other.mat
        return self

    def __add__(self, other):
        _new = CovMat(self.mat)
        _new += other
        return _new

    def __eq__(self, other):
        return np.all(self._mat == other.mat) if isinstance(other, CovMat) \
            else np.all(self._mat == other)

    def __len__(self):
        return self._size

    # -- private methods

    def _invalidate_cache(self):
        self._chol = None
        self._inverse = None
        self._cor_mat = None
        self._cond = None
        self._inverse = None

    # -- public interface

    def rescale(self, old_reference_values, new_reference_values):
        """
        Rescale the covariance matrix to new reference values.
        """
        _old_outer = np.asarray(np.outer(old_reference_values, old_reference_values))
        _new_outer = np.asarray(np.outer(new_reference_values, new_reference_values))
        self._mat *= _new_outer / _old_outer
        self._invalidate_cache()

    @property
    def mat(self):
        """
        Get the covariance matrix.
        """
        return np.array(self._mat)

    @mat.setter
    def mat(self, matrix):
        """
        Set the covariance matrix.
        """
        if isinstance(matrix, CovMat):
            matrix = matrix.mat

        self._mat = np.array(matrix)
        if self._mat.ndim != 2 or self._mat.shape[0] != self._mat.shape[1]:
            raise ValueError(
                "Covariance matrix must be square matrix, shape %r given." % (self._mat.shape,))
        self._size = self._mat.shape[0]
        self._cond = None

        self._invalidate_cache()

    @property
    def cor_mat(self):
        """
        Correlation matrix corresponding to the covariance matrix.
        """
        if self._cor_mat is None:
            _sqrt_vars = np.sqrt(np.diag(self._mat))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._cor_mat = self._mat / np.outer(_sqrt_vars, _sqrt_vars)
        return self._cor_mat

    @property
    def I(self):
        """
        Inverse of the covariance matrix. Returns ``None`` if matrix is singular.
        """
        if self._inverse is None:
            try:
                self._inverse = np.linalg.inv(self._mat)
            except np.linalg.LinAlgError:
                pass  # fail silently if matrix is singular
        return self._inverse

    @property
    def chol(self):
        """
        Lower diagonal matrix resulting from the Cholesky decomposition of the covariance matrix.
        Returns ``None`` if matrix is not positive definite.
        """
        if self._chol is None:
            try:
                self._chol = np.linalg.cholesky(self._mat)
            except np.linalg.LinAlgError:
                pass  # fail silently if matrix is not positive definite
        return self._chol

    @property
    def cond(self):
        """
        Condition number of the matrix.
        """
        if self._cond is None:
            self._cond = np.linalg.cond(self._mat)
        return self._cond

    @property
    def split_svd(self):
        if self.chol is None:
            return None
        _l = []
        _u, _v, _w = np.linalg.svd(self.chol)
        for _sv, _sc in zip(_v, _u.T):
            _l.append(np.outer(_sc, _sc) * _sv**2)
        return _l

    @property
    def split_diag_svd(self):
        _m0 = np.diag(np.diag(self._mat))
        _m = CovMat(self._mat - _m0)
        if _m is None:
            return None
        _l = [_m0]
        _u, _v, _w = np.linalg.svd(self.chol)
        for _sv, _sc in zip(_v, _u.T):
            _l.append(np.outer(_sc, _sc) * _sv ** 2)
        return _l


# Data structures for Gaussian Errors
@six.add_metaclass(abc.ABCMeta)
class GaussianErrorBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    @property
    @abc.abstractmethod
    def error(self):
        """Pointwise error array."""

    @property
    @abc.abstractmethod
    def error_rel(self):
        """Pointwise error array (relative errors)."""

    @property
    def reference(self):
        """Array of reference values for the error."""
        if callable(self._reference):
            return self._reference()
        else:
            return self._reference

    @reference.setter
    def reference(self, reference):
        if reference is None:
            ##self._reference = np.ones_like(self._err_val)
            self._reference = None
        elif callable(reference):
            self._reference = reference
        else:
            _ref = np.asarray(reference, dtype=float)
            # check for zero-valued references if error is marked 'relative'
            if self.relative and np.any(_ref == 0):
                warnings.warn(
                    "Relative error has a reference that contains values equal to zero: %s"
                    % _ref
                )
            self._reference = _ref

        # invalidate error_structures opposite declared relativity type
        if self.relative:
            self._cov_mat = None
            self._err = None
        else:
            self._cov_mat_rel = None
            self._err_rel = None

    @property
    @abc.abstractmethod
    def cov_mat(self):
        """Full absolute covariance matrix for error."""

    @property
    @abc.abstractmethod
    def cov_mat_rel(self):
        """Full relative covariance matrix for error."""

    @property
    @abc.abstractmethod
    def cor_mat(self):
        """Correlation matrix for error."""

    # TODO: remove _uncor/_cor from base interface?

    @property
    @abc.abstractmethod
    def error_uncor(self):
        """Pointwise array of 'uncorrelated' parts of absolute errors."""

    @property
    @abc.abstractmethod
    def error_cor(self):
        """Pointwise array of 'correlated' parts of absolute errors."""

    @property
    @abc.abstractmethod
    def error_rel_uncor(self):
        """Pointwise array of 'uncorrelated' parts of relative errors."""

    @property
    @abc.abstractmethod
    def error_rel_cor(self):
        """Pointwise array of 'correlated' parts of relative errors."""

    @property
    @abc.abstractmethod
    def cov_mat_uncor(self):
        """'Uncorrelated' part of absolute covariance matrix for error."""

    @property
    @abc.abstractmethod
    def cov_mat_cor(self):
        """'Fully correlated' part of absolute covariance matrix for error."""

    @property
    @abc.abstractmethod
    def cov_mat_rel_uncor(self):
        """'Uncorrelated' part of relative covariance matrix for error."""

    @property
    @abc.abstractmethod
    def cov_mat_rel_cor(self):
        """'Fully correlated' part of relative covariance matrix for error."""

    @property
    @abc.abstractmethod
    def fit_indices(self):
        """Indices of fits that have this error when used inside a MultiFit."""

    def get_cov_mat_object(self):
        """
        Returns the internally used `CovMat` object used to represent measurement errors. (advanced)
        """
        return self._cov_mat


class SimpleGaussianError(GaussianErrorBase):
    """
    A Gaussian Error constructed from an array of uncertainties, one for each data point, and a single
    non-negative correlation coefficient indicating the correlation of any two data points.
    This object constructs a covariance matrix object (py:obj:`CovMat`) from this information.

    An error object may be constructed as 'absolute' (default) or 'relative'. In the latter case, the
    uncertainty values are assumed to be given relative to the reference values (measurement or theory).
    There can be optionally specified using the 'reference' property.

    If an error object is declared as 'absolute' ('relative') and no 'reference' is set, then only 'absolute'
    ('relative') error arrays and covariance matrices are available. If 'reference' is set, these values are
    used to convert 'absolute' ('relative') error arrays or covariance matrices to 'relative' ('absolute') ones.

    """
    def __init__(self, err_val, corr_coeff, relative=False, reference=None, fit_indices=None):
        if not (0.0 <= corr_coeff <= 1.0):
            raise ValueError("Correlation must be between 0 and 1, %g given," % (corr_coeff,))
        self._corr_coeff = float(corr_coeff)
        self._is_relative = relative
        self.reference = reference
        self._fit_indices = fit_indices
        if self.relative:
            self.error_rel = err_val
        else:
            self.error = err_val

    # -- static methods

    @staticmethod
    def _calculate_cov_mat_generic(error_array, corr_coeff):
        """Calculate a covariance matrix from an array of error values and a global correlation coefficient."""
        if corr_coeff > 0:
            cov_mat_uncor_part = np.diag(error_array ** 2 * (1.0 - corr_coeff))
            cov_mat_cor_part = np.outer(error_array, error_array) * corr_coeff
        else:
            cov_mat_uncor_part = np.diag(error_array ** 2)
            cov_mat_cor_part = np.zeros_like(cov_mat_uncor_part)

        cov_mat = CovMat(cov_mat_uncor_part + cov_mat_cor_part)

        return cov_mat, cov_mat_uncor_part, cov_mat_cor_part

    # -- private methods

    def _calculate_cov_mat(self):
        """Calculate absolute covariance matrix for error object."""
        if self.relative:
            if self.reference is None:
                raise AttributeError("Requested 'absolute' errors for error object declared 'relative', but 'reference' not set!")
            _abs_err = self.error_rel * self.reference
        else:
            _abs_err = self.error

        self._cov_mat, self._cov_mat_uncor_part, self._cov_mat_cor_part = self._calculate_cov_mat_generic(_abs_err, self._corr_coeff)

    def _calculate_cov_mat_rel(self):
        """Calculate relative covariance matrix for error object."""
        _rel_err = self.error_rel
        self._cov_mat_rel, self._cov_mat_rel_uncor_part, self._cov_mat_rel_cor_part \
            = self._calculate_cov_mat_generic(_rel_err, self._corr_coeff)

    # -- public methods

    @property
    def relative(self):
        """Returns ``True`` if error is marked as a relative error."""
        return self._is_relative

    # @relative.setter
    # def relative(self, relative):
    #     """Set to ``True`` to mark as a relative error."""
    #     self._is_relative = relative

    @property
    def error(self):
        # calculate relative error from absolute error (if 'reference' is set)
        if self.relative:
            if self.reference is None:
                raise AttributeError(
                    "Requested 'absolute' errors for error object declared 'relative', but 'reference' not set!")
            self._err = self._err_rel * np.abs(self.reference)
        return self._err

    @error.setter
    def error(self, err_val):
        err_val = np.array(err_val, dtype=float)
        if np.any(err_val < 0):
            raise ValueError("Error values must be >= 0. Received: %s" % err_val)
        if self.relative:
            if self.reference is None:
                raise AttributeError(
                    "Setting 'absolute' errors for error object declared 'relative', but 'reference' not set!")

            self._err = err_val
            self._err_rel = err_val / np.abs(self.reference)
        else:
            self._err = err_val
            self._err_rel = None

        # invalidate cov mats
        self._cov_mat = None
        self._cov_mat_rel = None

    @property
    def error_uncor(self):
        # TODO: cache
        return self.error * np.sqrt(1.0 - self._corr_coeff)

    @property
    def error_cor(self):
        # TODO: cache
        return self.error * np.sqrt(self._corr_coeff)

    @property
    def error_rel_cor(self):
        # TODO: cache
        return self.error_rel * np.sqrt(self._corr_coeff)

    @property
    def error_rel_uncor(self):
        # TODO: cache
        return self.error_rel * np.sqrt(1.0 - self._corr_coeff)

    @property
    def error_rel(self):
        # calculate relative error from absolute error (if 'reference' is set)
        if not self.relative:
            if self.reference is None:
                raise AttributeError(
                    "Requested 'relative' errors for error object declared 'absolute', but 'reference' not set!")
            self._err_rel = self._err / np.abs(self.reference)
        return self._err_rel

    @error_rel.setter
    def error_rel(self, err_val):
        err_val = np.array(err_val, dtype=float)
        if np.any(err_val < 0):
            raise ValueError("Error values must be >= 0. Received: %s" % err_val)
        if self.relative:
            self._err = None
            self._err_rel = err_val
        else:
            if self.reference is None:
                raise AttributeError(
                    "Setting 'absolute' errors for error object declared 'relative', but 'reference' not set!")
            self._err = err_val * self.reference
            self._err_rel = err_val

        # invalidate cov mats
        self._cov_mat = None
        self._cov_mat_rel = None

    @property
    def cov_mat(self):
        if self._cov_mat is None:
            self._calculate_cov_mat()
        return self._cov_mat.mat

    @property
    def cov_mat_inverse(self):
        if self._cov_mat is None:
            self._calculate_cov_mat()
        return self._cov_mat.I

    @property
    def cov_mat_uncor(self):
        if self._cov_mat is None:
            self._calculate_cov_mat()
        return self._cov_mat_uncor_part

    @property
    def cov_mat_cor(self):
        if self._cov_mat is None:
            self._calculate_cov_mat()
        return self._cov_mat_cor_part

    @property
    def cov_mat_rel(self):
        if self._cov_mat_rel is None:
            self._calculate_cov_mat_rel()
        return self._cov_mat_rel.mat

    @property
    def cov_mat_rel_inverse(self):
        if self._cov_mat is None:
            self._calculate_cov_mat_rel()
        return self._cov_mat_rel.I

    @property
    def cov_mat_rel_uncor(self):
        if self._cov_mat_rel is None:
            self._calculate_cov_mat_rel()
        return self._cov_mat_rel_uncor_part

    @property
    def cov_mat_rel_cor(self):
        if self._cov_mat_rel is None:
            self._calculate_cov_mat_rel()
        return self._cov_mat_rel_cor_part

    @property
    def cor_mat(self):
        if self.relative:
            _ = self.cov_mat_rel  # call to calculate CovMat
            return self._cov_mat_rel.cor_mat
        else:
            _ = self.cov_mat  # call to calculate CovMat
            return self._cov_mat.cor_mat

    @property
    def corr_coeff(self):
        return self._corr_coeff

    @property
    def fit_indices(self):
        return self._fit_indices


class MatrixGaussianError(GaussianErrorBase):
    """
    A Gaussian Error constructed from a covariance matrix, or a correlation matrix together with
    a pointwise error array.

    An error object may be constructed as 'absolute' (default) or 'relative'. In the latter case, the
    uncertainty values are assumed to be given relative to the reference values (measurement or theory).
    There can be optionally specified using the 'reference' property.

    If an error object is declared as 'absolute' ('relative') and no 'reference' is set, then only 'absolute'
    ('relative') error arrays and covariance matrices are available. If 'reference' is set, these values are
    used to convert 'absolute' ('relative') error arrays or covariance matrices to 'relative' ('absolute') ones.

    """
    def __init__(self, err_matrix, matrix_type, err_val=None, relative=False,
                 reference=None, fit_indices=None):
        self._is_relative = relative
        self.reference = reference
        self._fit_indices = fit_indices

        if err_val is not None:
            err_val = np.asarray(err_val)

        self._err = None
        self._err_rel = None
        self._cov_mat = None
        self._cov_mat_rel = None

        # set the main matrix
        if matrix_type.lower() in ('covariance', 'cov'):
            if err_val is not None:
                raise ValueError("Cannot provide err_val when constructing a covariance matrix!")
            self._matrix_type_at_construction = 'covariance'
            if self.relative:
                self.cov_mat_rel = err_matrix
            else:
                self.cov_mat = err_matrix
        elif matrix_type.lower() in ('correlation', 'correlations', 'cor', 'corr'):
            self._matrix_type_at_construction = 'correlation'
            if err_val is None:
                raise ValueError(
                    "Cannot construct matrix-type error from correlation matrix "
                    "without an array of error values!")
            _cm = self._calculate_cov_mat_from_cor_mat_and_error_array(err_val, err_matrix)
            if self.relative:
                self.cov_mat_rel = _cm
            else:
                self.cov_mat = _cm
        else:
            raise ValueError("Unknown matrix type '%s'. Expected one of: %r" % (matrix_type, ('cov', 'cor')))

    # -- static methods

    @staticmethod
    def _calculate_cov_mat_from_cor_mat_and_error_array(error_array, corr_mat):
        """Calculate a covariance matrix from an array of error values and a correlation matrix."""
        # check if corr_mat has ones on diagonal
        if not np.allclose(np.diag(corr_mat), 1.0):
            raise ValueError("Corelation matrix has non-unit entry on diagonal!")
        # TODO: check if corr_mat is symmetric and positive definite (?)
        cov_mat = np.asarray(np.outer(error_array, error_array)) * np.asarray(corr_mat)
        return CovMat(cov_mat)

    @staticmethod
    def _calculate_cov_mat_rel_from_cov(cov_mat, reference):
        _ref = np.asarray(reference)
        _refmat = np.outer(_ref, _ref)
        _mat = np.asarray(cov_mat)
        return CovMat(_mat / _refmat)

    @staticmethod
    def _calculate_cov_mat_from_cov_rel(cov_mat_rel, reference):
        _ref = np.asarray(reference)
        _refmat = np.outer(_ref, _ref)
        _mat_rel = np.asarray(cov_mat_rel)
        return CovMat(_mat_rel * _refmat)

    # -- public methods

    @property
    def relative(self):
        """Returns ``True`` if error is marked as a relative error."""
        return self._is_relative

    @property
    def cov_mat(self):
        """"""
        if self.relative:
            if self.reference is None:
                raise AttributeError(
                    "Requested 'absolute' covariance matrix for error object declared 'relative', "
                    "but 'reference' not set!")
            self._cov_mat = self._calculate_cov_mat_from_cov_rel(self.cov_mat_rel, self.reference)
        return self._cov_mat.mat

    @cov_mat.setter
    def cov_mat(self, cov_mat):
        """"""
        self._cov_mat = CovMat(cov_mat)
        self._cov_mat_rel = None

    @property
    def cov_mat_rel(self):
        """"""
        if not self.relative:
            if self.reference is None:
                raise AttributeError(
                    "Requested 'relative' covariance matrix for error object declared 'absolute', but 'reference' not set!")
            self._cov_mat_rel = self._calculate_cov_mat_rel_from_cov(self.cov_mat, self.reference)
        return self._cov_mat_rel.mat

    @cov_mat_rel.setter
    def cov_mat_rel(self, cov_mat_rel):
        """"""
        self._cov_mat_rel = CovMat(cov_mat_rel)
        self._cov_mat = None

    @property
    def cov_mat_inverse(self):
        _ = self.cov_mat  # call to initialize
        return self._cov_mat.I

    @property
    def error(self):
        """"""
        if self._err is None:
            if self.relative:
                if self.reference is None:
                    raise AttributeError(
                        "Requested 'absolute' error array for error object declared 'relative', but 'reference' not set!")
            self._err = np.sqrt(np.diag(self.cov_mat))
        return self._err

    @property
    def error_rel(self):
        """"""
        if self._err_rel is None:
            if not self.relative:
                if self.reference is None:
                    raise AttributeError(
                        "Requested 'relative' error array for error object declared 'absolute', but 'reference' not set!")
            self._err_rel = np.sqrt(np.diag(self.cov_mat_rel))
        return self._err_rel

    @property
    def error_uncor(self):
        raise AttributeError("Cannot get the uncorrelated part of a 'matrix-type' error!")

    @property
    def error_cor(self):
        raise AttributeError("Cannot get the correlated part of a 'matrix-type' error!")

    @property
    def error_rel_cor(self):
        raise AttributeError("Cannot get the correlated part of a 'matrix-type' error!")

    @property
    def error_rel_uncor(self):
        raise AttributeError("Cannot get the uncorrelated part of a 'matrix-type' error!")

    @property
    def cov_mat_uncor(self):
        raise AttributeError("Cannot get the uncorrelated part of a 'matrix-type' error!")

    @property
    def cov_mat_cor(self):
        raise AttributeError("Cannot get the correlated part of a 'matrix-type' error!")

    @property
    def cov_mat_rel_uncor(self):
        raise AttributeError("Cannot get the uncorrelated part of a 'matrix-type' error!")

    @property
    def cov_mat_rel_cor(self):
        raise AttributeError("Cannot get the correlated part of a 'matrix-type' error!")

    @property
    def cor_mat(self):
        _ = self.cov_mat  # call to initialize
        return self._cov_mat.cor_mat

    @property
    def fit_indices(self):
        return self._fit_indices

    # For performance reasons this check is called manually from outside
    def check_cov_mat_symmetry(self):
        _mat = self.cov_mat
        if np.any(_mat != _mat.T):
            warnings.warn("Covariance matrix is not symmetrical: %s" % _mat)

