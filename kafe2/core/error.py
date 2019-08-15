"""
# TODO: make all setters copy and own data members
"""

import abc
import copy
import numpy as np
import six

import logging

logger = logging.getLogger(__name__)
logging.basicConfig()


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
    return CovMat(np.asmatrix(_mat))


"""
Data structure for Covariance Matrices
"""


class CovMat(object):
    def __init__(self, matrix):
        # -- member definitions
        self._mat = None
        self._size = None
        self._inverse = None
        self._chol = None
        self._cor_mat = None
        self._cond = None

        # -- initialization
        self.mat = matrix

    # -- 'magic' methods

    def __iadd__(self, other):
        self.mat += other.mat
        return self

    def __add__(self, other):
        _new = CovMat(self.mat)
        _new += other
        return _new

    def __len__(self):
        return self._size

    # -- private methods

    def _invalidate_cache(self):
        self._chol = None
        self._inverse = None
        self._cor_mat = None

    # -- public interface

    def rescale_variant(self, old_reference_values, new_reference_values):
        """
        Rescale the covariance matrix (variant implementation, pure Python).
        """
        for i in six.moves.range(self._size):
            for j in six.moves.range(self._size):
                _v = self._mat[i, j]
                _v /= old_reference_values[i]
                _v /= old_reference_values[j]
                _v *= new_reference_values[i]
                _v *= new_reference_values[j]
                self._mat[i, j] = _v

        self._invalidate_cache()

    def rescale(self, old_reference_values, new_reference_values):
        """
        Rescale the covariance matrix.
        """
        _old_outer = np.asarray(np.outer(old_reference_values, old_reference_values))
        _new_outer = np.asarray(np.outer(new_reference_values, new_reference_values))
        self._mat /= _old_outer / _new_outer
        self._invalidate_cache()

    @property
    def mat(self):
        """
        Get the covariance matrix.
        """
        return self._mat

    @mat.setter
    def mat(self, matrix):
        """
        Set the covariance matrix.
        """
        if isinstance(matrix, CovMat):
            # "copy constructor"
            matrix = copy.deepcopy(matrix.mat)

        self._mat = np.asmatrix(matrix)
        if not (self._mat.shape[1] == self._mat.shape[0]):
            raise ValueError("Covariance matrix must be square matrix, shape %r given," % (self._mat.shape,))
        if not np.allclose(self._mat - self._mat.T, 0):
            raise ValueError("Covariance matrix must be symmetric!")
        self._size = self._mat.shape[0]
        self._cond = None

        self._invalidate_cache()

    @property
    def cor_mat(self):
        """
        Correlation matrix corresponding to the covariance matrix.
        """
        if self._cor_mat is None:
            _sqrt_vars = np.sqrt(np.diag(self.mat))
            _mat_as_arr = np.asarray(self.mat)
            self._cor_mat = np.asmatrix(_mat_as_arr / np.outer(_sqrt_vars, _sqrt_vars))
        return self._cor_mat

    @property
    def I(self):
        """
        Inverse of the covariance matrix. Returns ``None`` if matrix is singular.
        """
        if self._inverse is None:
            try:
                self._inverse = self._mat.I
            except np.linalg.LinAlgError:
                pass  # fail silently if matrix is singular
        return self._inverse

    @property
    def chol(self):
        """
        Lower diagonal matrix resulting from the Cholesky decomposition of the covariance matrix. Returns ``None``
        if matrix is not positive definite.
        """
        if self._chol is None:
            try:
                self._chol = np.linalg.cholesky(self.mat)
            except np.linalg.LinAlgError:
                pass  # fail silently if matrix is not positive definite
        return self._chol

    @property
    def cond(self):
        """
        Condition number of the matrix.
        """
        if self._cond is None:
            self._cond = np.linalg.cond(self.mat)
        return self._cond

    def split_svd(self):
        if self.chol is None:
            return None
        _l = []
        _u, _v, _w = np.linalg.svd(self.chol)
        for _sv, _sc in zip(_v, _u.T):
            _l.append(np.outer(_sc, _sc) * _sv**2)
        return _l

    def split_diag_svd(self):
        _m0 = np.diag(np.diag(self.mat))
        _m = CovMat(self.mat - _m0)
        if _m  is None:
            return None
        _l = [_m0]
        _u, _v, _w = np.linalg.svd(self.chol)
        for _sv, _sc in zip(_v, _u.T):
            _l.append(np.outer(_sc, _sc) * _sv**2)
        return _l


"""
Data structures for Gaussian Errors
"""


@six.add_metaclass(abc.ABCMeta)
class GaussianErrorBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """

    @abc.abstractproperty
    def error(self):
        """Pointwise error array."""
        pass

    @abc.abstractproperty
    def error_rel(self):
        """Pointwise error array (relative errors)."""
        pass

    @abc.abstractproperty
    def reference(self):
        """Array of reference values for the error."""
        pass

    @abc.abstractproperty
    def cov_mat(self):
        """Full absolute covariance matrix for error."""
        pass

    @abc.abstractproperty
    def cov_mat_rel(self):
        """Full relative covariance matrix for error."""
        pass

    @abc.abstractproperty
    def cor_mat(self):
        """Correlation matrix for error."""
        pass

    # TODO: remove _uncor/_cor from base interface?

    @abc.abstractproperty
    def error_uncor(self):
        """Pointwise array of 'uncorrelated' parts of absolute errors."""
        pass

    @abc.abstractproperty
    def error_cor(self):
        """Pointwise array of 'correlated' parts of absolute errors."""
        pass

    @abc.abstractproperty
    def error_rel_uncor(self):
        """Pointwise array of 'uncorrelated' parts of relative errors."""
        pass

    @abc.abstractproperty
    def error_rel_cor(self):
        """Pointwise array of 'correlated' parts of relative errors."""
        pass

    @abc.abstractproperty
    def cov_mat_uncor(self):
        """'Uncorrelated' part of absolute covariance matrix for error."""
        pass

    @abc.abstractproperty
    def cov_mat_cor(self):
        """'Fully correlated' part of absolute covariance matrix for error."""
        pass

    @abc.abstractproperty
    def cov_mat_rel_uncor(self):
        """'Uncorrelated' part of relative covariance matrix for error."""
        pass

    @abc.abstractproperty
    def cov_mat_rel_cor(self):
        """'Fully correlated' part of relative covariance matrix for error."""
        pass

    def get_cov_mat_object(self):
        """Returns the internal-use `CovMat` object used to represent measurement errors. (advanced)"""
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
    def __init__(self, err_val, corr_coeff, relative=False, reference=None):
        if not (0.0 <= corr_coeff <= 1.0):
            raise ValueError("Correlation must be between 0 and 1, %g given," % (corr_coeff,))
        self._corr_coeff = float(corr_coeff)
        self._is_relative = relative
        self.reference = reference
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

        assert np.allclose(np.diag(cov_mat.mat), error_array ** 2, atol=1e-4)

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
        if self.relative:
            _rel_err = self.error_rel
        else:
            if self.reference is None:
                raise AttributeError("Requested 'relative' errors for error object declared 'absolute', but 'reference' not set!")
            _rel_err = self.error / self.reference

        self._cov_mat_rel, self._cov_mat_rel_uncor_part, self._cov_mat_rel_cor_part = self._calculate_cov_mat_generic(_rel_err, self._corr_coeff)

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
            self._err = self._err_rel * self.reference
        return self._err

    @error.setter
    def error(self, err_val):
        err_val = np.asarray(err_val, dtype=float)
        if self.relative:
            if self.reference is None:
                raise AttributeError(
                    "Setting 'absolute' errors for error object declared 'relative', but 'reference' not set!")

            self._err = err_val
            self._err_rel = err_val / self.reference
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
            self._err_rel = self._err / self.reference
        return self._err_rel

    @error_rel.setter
    def error_rel(self, err_val):
        err_val = np.asarray(err_val, dtype=float)
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
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        if reference is None:
            ##self._reference = np.ones_like(self._err_val)
            self._reference = None
        else:
            _ref = np.asarray(reference, dtype=float)
            # check for zero-valued references if error is marked 'relative'
            if self.relative and np.any(np.isclose(_ref, 0.0, atol=1e-5)):
                # TODO: avoid hard-coded value
                print("WARNING: Error reference contains zero values! Replacing them "
                      "with default minimum '%g'" % (1e-2,))
                _default = np.ones_like(_ref) * 1e-2
                _ref = np.where(np.isclose(_ref, 0.0, atol=1e-5), _default, _ref)
            self._reference = _ref

        # invalidate error_structures opposite declared relativity type
        if self.relative:
            self._cov_mat = None
            self._err = None
        else:
            self._cov_mat_rel = None
            self._err_rel = None


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
            self._calculate_cov_mat()
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
            return self.cov_mat_rel.cor_mat
        else:
            return self.cov_mat.cor_mat

    @property
    def corr_coeff(self):
        return self._corr_coeff

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
    def __init__(self, err_matrix, matrix_type, err_val=None, relative=False, reference=None):
        self._is_relative = relative
        self.reference = reference

        if err_val is not None:
            err_val = np.asarray(err_val)

        self._err = None
        self._err_rel = None
        self._cov_mat = None
        self._cov_mat_rel = None

        # set the main matrix
        if matrix_type.lower() in ('covariance', 'cov'):
            self._matrix_type_at_construction = 'covariance'
            if self.relative:
                self.cov_mat_rel = err_matrix
                # check err_val against cov_mat diagonal
                if err_val is not None:
                    if not np.allclose(np.diag(self._cov_mat_rel.mat), err_val ** 2):
                        raise ValueError("Covariance matrix diagonal does not match array of error values!")
            else:
                self.cov_mat = err_matrix
                # check err_val against cov_mat diagonal
                if err_val is not None:
                    if not np.allclose(np.diag(self._cov_mat.mat), err_val ** 2):
                        raise ValueError("Covariance matrix diagonal does not match array of error values!")

        elif matrix_type.lower() in ('correlation', 'correlations', 'cor', 'corr'):
            self._matrix_type_at_construction = 'correlation'
            if err_val is None:
                raise ValueError("Cannot construct matrix-type error from correlation matrix without an array of error values!")
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
                    "Requested 'absolute' covariance matrix for error object declared 'relative', but 'reference' not set!")
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
        if self.relative:
            if self.reference is None:
                raise AttributeError(
                    "Requested 'absolute' inverse covariance matrix for error object declared 'relative', but 'reference' not set!")
            self._cov_mat = self._calculate_cov_mat_from_cov_rel(self.cov_mat_rel, self.reference)
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
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        if reference is None:
            self._reference = None
        else:
            _ref = np.asarray(reference, dtype=float)
            # check for zero-valued references if error is marked 'relative'
            if self.relative and np.any(np.isclose(_ref, 0.0, atol=1e-5)):
                # TODO: avoid hard-coded value
                print("WARNING: Error reference contains zero values! Replacing them "
                      "with default minimum '%g'" % (1e-2,))
                _default = np.ones_like(_ref) * 1e-2
                _ref = np.where(np.isclose(_ref, 0.0, atol=1e-5), _default, _ref)
            self._reference = _ref

        # invalidate error_structures opposite declared relativity type
        if self.relative:
            self._cov_mat = None
            self._err = None
        else:
            self._cov_mat_rel = None
            self._err_rel = None


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
        # TODO: check if these are equal
        if self.relative:
            return self._cov_mat_rel.cor_mat
        else:
            return self._cov_mat.cor_mat




if __name__ == '__main__':
    pass


