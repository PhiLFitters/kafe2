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
    def __init__(self, indices, values, cov_mat, relative=False):
        self._indices = np.array(indices)
        self._values = np.array(values)

        _cov_mat_array = np.array(cov_mat)
        if not np.array_equal(_cov_mat_array, _cov_mat_array.T):
            raise ParameterConstraintException('The covariance matrix for parameter constraints must be symmetric!')
        if len(self._values.shape) != 1 or self._values.shape * 2 != _cov_mat_array.shape:
            raise ParameterConstraintException(
                'Expected values and cov_mat to be of shapes (N, ), (N, N) but received shapes %s, %s instead!'
                % (self._values.shape, _cov_mat_array.shape))

        if relative:
            self._cov_mat_abs = None
            self._cov_mat_rel = _cov_mat_array
        else:
            self._cov_mat_abs = _cov_mat_array
            self._cov_mat_rel = None
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
            self._cov_mat_abs = self._cov_mat_rel * np.outer(self._values, self._values)
        return self._cov_mat_abs

    @property
    def cov_mat_rel(self):
        if self._cov_mat_rel is None:
            self._cov_mat_rel = self._cov_mat_abs / np.outer(self._values, self._values)
        return self._cov_mat_rel

    @property
    def relative(self):
        return self._relative

    @property
    def cov_mat_inverse(self):
        if self._cov_mat_inverse is None:
            self._cov_mat_inverse = np.linalg.inv(self.cov_mat)
        return self._cov_mat_inverse
