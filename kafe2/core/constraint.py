import abc
import numpy as np
import six

from ..fit.io.file import FileIOMixin


class ParameterConstraintException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class ParameterConstraint(FileIOMixin, object):
    """
    Abstract base class for parameter constraints.
    Subclasses must implement the ``cost`` method.
    """

    def __init__(self):
        pass

    def _get_base_class(self):
        return ParameterConstraint

    def _get_object_type_name(self):
        return 'parameter_constraint'

    def cost(self, parameter_values):
        """
        Calculates additional cost depending on the fit parameter values.

        :param parameter_values: The current parameter values of the fit
        :type parameter_values: iterable of float
        :return: The additional cost imposed by the given parameter values
        :rtype: float
        """
        pass


class GaussianSimpleParameterConstraint(ParameterConstraint):

    def __init__(self, index, value, uncertainty, relative=False):
        """
        Simple class for applying a gaussian constraint to a single parameter of a fit.

        :param index: The index of the parameter to be constrained
        :type index: int
        :param value: The value to which the parameter should be constrained
        :type value: float
        :param uncertainty: The uncertainty with which the parameter should be constrained to the given value
        :type uncertainty: float
        :param relative: Whether the given uncertainty is relative to the given value
        :type relative: bool
        """
        self._index = index
        self._value = value
        if relative:
            self._uncertainty_abs = None
            self._uncertainty_rel = uncertainty
        else:
            self._uncertainty_abs = uncertainty
            self._uncertainty_rel = None
        self._relative = relative
        super(GaussianSimpleParameterConstraint, self).__init__()

    @property
    def index(self):
        """the index of the constrained parameter"""
        return self._index

    @property
    def value(self):
        """the value to which the parameter is being constrained"""
        return self._value

    @property
    def uncertainty(self):
        """the absolute uncertainty with which the parameter is being constrained"""
        if self._uncertainty_abs is None:
            self._uncertainty_abs = self._uncertainty_rel * self.value
        return self._uncertainty_abs

    @property
    def uncertainty_rel(self):
        """the uncertainty relative to ``value`` with which the parameter is being constrained"""
        if self._uncertainty_rel is None:
            self._uncertainty_rel = self._uncertainty_abs / self.value
        return self._uncertainty_rel

    @property
    def relative(self):
        """whether the constraint was initialized with a relative uncertainty"""
        return self._relative

    def cost(self, parameter_values):
        """
        Calculates additional cost depending on the fit parameter values.
        More specifically, the constraint first picks the value from ``parameter_values`` at ``self.index``.
        The constraint then calculates the residual by subtracting ``self.value``.
        The final cost is calculated by dividing the residual by ``self.uncertainty`` and squaring the result.

        :param parameter_values: The current parameter values of the fit
        :type parameter_values: iterable of float
        :return: The additional cost imposed by the given parameter values
        :rtype: float
        """
        return ((parameter_values[self.index] - self.value) / self.uncertainty) ** 2


class GaussianMatrixParameterConstraint(ParameterConstraint):

    def __init__(self, indices, values, matrix, matrix_type='cov', uncertainties=None, relative=False):
        """
        Advanced class for applying correlated constraints to several parameters of a fit.
        The order of ``indices``, ``values``, ``matrix``, and ``uncertainties`` must be aligned.
        In other words the first index must belong to the first value, the first row/column in the matrix, etc.

        Let N be the number of parameters to be constrained.
        :param indices: The indices of the parameters to be constrained
        :type indices: iterable of int, shape (N,)
        :param values: The values to which the parameters should be constrained
        :type values: iterable of float, shape (N,)
        :param matrix: The matrix that defines the correlation between the parameters. By default interpreted as a
            covariance matrix. Can also be interpreted as a correlation matrix by setting ``matrix_type``
        :type matrix: iterable of float, shape (N, N)
        :param matrix_type: Whether the matrix should be interpreted as a covariance matrix or as a correlation matrix
        :type matrix_type: str, either 'cov' or 'cor'
        :param uncertainties: The uncertainties to be used in conjunction with a correlation matrix
        :type uncertainties: ``None`` or iterable of float, shape (N,)
        :param relative: Whether the covariance matrix/the uncertainties should be interpreted as relative to ``values``
        :type relative: bool
        """
        self._indices = np.array(indices)
        self._values = np.array(values)

        _matrix_array = np.array(matrix)
        if not np.array_equal(_matrix_array, _matrix_array.T):
            raise ValueError('The matrix for parameter constraints must be symmetric!')
        if len(self._values.shape) != 1 or self._values.shape * 2 != _matrix_array.shape:
            raise ValueError(
                'Expected values and cov_mat to be of shapes (N, ), (N, N) but received shapes %s, %s instead!'
                % (self._values.shape, _matrix_array.shape))
        if matrix_type == 'cov':
            pass
        elif matrix_type == 'cor':
            if np.any(np.diag(_matrix_array) != 1.0):
                raise ValueError('The correlation matrix has diagonal elements that aren\'t equal to 1!')
            if np.any(_matrix_array> 1.0):
                raise ValueError('The correlation matrix has elements greater than 1!')
            if np.any(_matrix_array < -1.0):
                raise ValueError('The correlation matrix has elements smaller than -1!')
        else:
            raise ValueError('Unknown matrix_type: %s, must be either cov or cor!' % matrix_type)

        if matrix_type == 'cov':
            if relative:
                self._cov_mat_abs = None
                self._cov_mat_rel = _matrix_array
                self._cor_mat = None
            else:
                self._cov_mat_abs = _matrix_array
                self._cov_mat_rel = None
                self._cor_mat = None

            if uncertainties is not None:
                raise ValueError('Uncertainties can only be specified if matrix_type is cov!')
            self._uncertainties_abs = None
            self._uncertainties_rel = None
        else:
            self._cov_mat_abs = None
            self._cov_mat_rel = None
            self._cor_mat = _matrix_array
            if uncertainties is None:
                raise ValueError('If matrix_type is cor uncertainties must be specified!')
            if relative:
                self._uncertainties_abs = None
                self._uncertainties_rel = uncertainties
            else:
                self._uncertainties_abs = uncertainties
                self._uncertainties_rel = None

        self._matrix_type = matrix_type
        self._relative = relative

        self._cov_mat_inverse = None
        super(GaussianMatrixParameterConstraint, self).__init__()

    @property
    def indices(self):
        """the indices of the parameters to be constrained"""
        return self._indices

    @property
    def values(self):
        """the values to which the parameters are being constrained"""
        return self._values

    @property
    def cov_mat(self):
        """the absolute covariance matrix between the parameter uncertainties"""
        if self._cov_mat_abs is None:
            if self.matrix_type == 'cov':
                self._cov_mat_abs = self._cov_mat_rel * np.outer(self.values, self.values)
            else:
                self._cov_mat_abs = self._cor_mat * np.outer(self.uncertainties, self.uncertainties)
        return self._cov_mat_abs

    @property
    def cov_mat_rel(self):
        """the covariance matrix between the parameter uncertainties relative to ``self.values``"""
        if self._cov_mat_rel is None:
            if self.matrix_type == 'cov':
                self._cov_mat_rel = self._cov_mat_abs / np.outer(self.values, self.values)
            else:
                self._cov_mat_rel = self._cor_mat * np.outer(self.uncertainties_rel, self.uncertainties_rel)
        return self._cov_mat_rel

    @property
    def cor_mat(self):
        """the correlation matrix between the parameter uncertainties"""
        if self._cor_mat is None:
            # if the originally specified cov mat was relative, calculate the cor mat based on that
            if self._relative:
                self._cor_mat = self.cov_mat_rel / np.outer(self.uncertainties_rel, self.uncertainties_rel)
            else:
                self._cor_mat = self.cov_mat / np.outer(self.uncertainties, self.uncertainties)
        return self._cor_mat

    @property
    def uncertainties(self):
        """the uncorrelated, absolute uncertainties for the parameters to be constrained to"""
        if self._uncertainties_abs is None:
            if self.matrix_type == 'cov':
                self._uncertainties_abs = np.sqrt(np.diag(self.cov_mat))
            else:
                self._uncertainties_abs = self.uncertainties_rel * self.values
        return self._uncertainties_abs

    @property
    def uncertainties_rel(self):
        """the uncorrelated uncertainties for the parameters to be constrained to relative to ``self.values``"""
        if self._uncertainties_rel is None:
            if self.matrix_type == 'cov':
                self._uncertainties_rel = np.sqrt(np.diag(self.cov_mat_rel))
            else:
                self._uncertainties_rel = self.uncertainties / self.values
        return self._uncertainties_rel

    @property
    def matrix_type(self):
        """the type of matrix with which the constraint was initialized"""
        return self._matrix_type

    @property
    def relative(self):
        """whether the constraint was initialized with a relative covariance matrix/with relative uncertainties"""
        return self._relative

    @property
    def cov_mat_inverse(self):
        """the inverse of the covariance matrix between the parameter uncertainties"""
        if self._cov_mat_inverse is None:
            self._cov_mat_inverse = np.linalg.inv(self.cov_mat)
        return self._cov_mat_inverse

    def cost(self, parameter_values):
        """
        Calculates additional cost depending on the fit parameter values.
        More specifically, the constraint first picks values from ``parameter_values`` according to ``self.indices``.
        The constraint then calculates the residuals by subtracting ``self.values``.
        The final cost is calculated by applying the residuals to both sides of ``self.cov_mat_inverse``
        via dot product.

        :param parameter_values: The current parameter values of the fit
        :type parameter_values: iterable of float
        :return: The additional cost imposed by the given parameter values
        :rtype: float
        """
        _selected_par_values = np.asarray(parameter_values)[self.indices]
        _res = _selected_par_values - self.values
        return _res.dot(self.cov_mat_inverse).dot(_res)
