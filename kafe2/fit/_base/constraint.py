from abc import ABCMeta
import numpy as np

from kafe2.fit.io.file import FileIOMixin


class ParameterConstraintException(Exception):
    pass


class ParameterConstraint(FileIOMixin, object, metaclass=ABCMeta):
    # TODO documentation

    def __init__(self):
        pass

    def _get_base_class(self):
        return ParameterConstraint

    def _get_object_type_name(self):
        return 'parameter_constraint'

    def cost(self, parameter_values):
        pass


class GaussianSimpleParameterConstraint(ParameterConstraint):
    # TODO documentation
    def __init__(self, index, value, uncertainty, relative=False):
        self._index = index
        self._value = value
        if relative:
            self._uncertainty_abs = None
            self._uncertainty_rel = uncertainty
        else:
            self._uncertainty_abs = uncertainty
            self._uncertainty_rel = None
        self._relative = relative
        super(GaussianSimpleParameterConstraint).__init__()

    def cost(self, parameter_values):
        return ((parameter_values[self.index] - self.value) / self.uncertainty) ** 2

    @property
    def index(self):
        return self._index

    @property
    def value(self):
        return self._value

    @property
    def uncertainty(self):
        if self._uncertainty_abs is None:
            self._uncertainty_abs = self._uncertainty_rel * self.value
        return self._uncertainty_abs

    @property
    def uncertainty_rel(self):
        if self._uncertainty_rel is None:
            self._uncertainty_rel = self._uncertainty_abs / self.value
        return self._uncertainty_rel

    @property
    def relative(self):
        return self._relative


class GaussianMatrixParameterConstraint(ParameterConstraint):
    # TODO documentation
    def __init__(self, indices, values, matrix, matrix_type='cov', uncertainties=None, relative=False):
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
        super(GaussianMatrixParameterConstraint).__init__()

    def cost(self, parameter_values):
        _selected_par_values = np.asarray(parameter_values)[self.indices]
        _res = _selected_par_values - self.values
        return _res.dot(self.cov_mat_inverse).dot(_res)

    @property
    def indices(self):
        return self._indices

    @property
    def values(self):
        return self._values

    @property
    def cov_mat(self):
        if self._cov_mat_abs is None:
            if self.matrix_type is 'cov':
                self._cov_mat_abs = self._cov_mat_rel * np.outer(self.values, self.values)
            else:
                self._cov_mat_abs = self._cor_mat * np.outer(self.uncertainties, self.uncertainties)
        return self._cov_mat_abs

    @property
    def cov_mat_rel(self):
        if self._cov_mat_rel is None:
            if self.matrix_type == 'cov':
                self._cov_mat_rel = self._cov_mat_abs / np.outer(self.values, self.values)
            else:
                self._cov_mat_rel = self._cor_mat * np.outer(self.uncertainties_rel, self.uncertainties_rel)
        return self._cov_mat_rel

    @property
    def cor_mat(self):
        if self._cor_mat is None:
            # if the originally specified cov mat was relative, calculate the cor mat based on that
            if self._relative:
                self._cor_mat = self.cov_mat_rel / np.outer(self.uncertainties_rel, self.uncertainties_rel)
            else:
                self._cor_mat = self.cov_mat / np.outer(self.uncertainties, self.uncertainties)
        return self._cor_mat

    @property
    def uncertainties(self):
        if self._uncertainties_abs is None:
            if self.matrix_type == 'cov':
                self._uncertainties_abs = np.sqrt(np.diag(self.cov_mat))
            else:
                self._uncertainties_abs = self.uncertainties_rel * self.values
        return self._uncertainties_abs

    @property
    def uncertainties_rel(self):
        if self._uncertainties_rel is None:
            if self.matrix_type == 'cov':
                self._uncertainties_rel = np.sqrt(np.diag(self.cov_mat_rel))
            else:
                self._uncertainties_rel = self.uncertainties / self.values
        return self._uncertainties_rel

    @property
    def matrix_type(self):
        return self._matrix_type

    @property
    def relative(self):
        return self._relative

    @property
    def cov_mat_inverse(self):
        if self._cov_mat_inverse is None:
            self._cov_mat_inverse = np.linalg.inv(self.cov_mat)
        return self._cov_mat_inverse
