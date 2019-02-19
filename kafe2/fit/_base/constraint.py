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

    def cost(self):
        pass


class GaussianSimpleParameterConstraint(ParameterConstraint):
    # TODO documentation
    def __init__(self, index, value, uncertainty):
        self._index = index
        self._value = value
        self._uncertainty = uncertainty
        super(GaussianSimpleParameterConstraint).__init__()

    def cost(self, parameter_values):
        return ((parameter_values[self._index] - self._value) / self._uncertainty) ** 2

    @property
    def index(self):
        return self._index

    @property
    def value(self):
        return self._value

    @property
    def uncertainty(self):
        return self._uncertainty

class GaussianMatrixParameterConstraint(ParameterConstraint):
    # TODO documentation
    def __init__(self, indices, values, cov_mat):
        self._indices = np.array(indices)
        self._values = np.array(values)
        self._cov_mat = np.array(cov_mat)
        if not np.array_equal(self._cov_mat, self._cov_mat.T):
            raise ParameterConstraintException('The covariance matrix for parameter constraints must be symmetric!')
        if len(self._values.shape) != 1 or self._values.shape * 2 != self._cov_mat.shape:
            raise ParameterConstraintException(
                'Expected values and cov_mat to be of shapes (N, ), (N, N) but received shapes %s, %s instead!'
                % (self._values.shape, self._cov_mat.shape))

        self._cov_mat_inverse = None
        super(GaussianMatrixParameterConstraint).__init__()

    @property
    def indices(self):
        return self._indices

    @property
    def values(self):
        return self._values

    @property
    def cov_mat(self):
        return self._cov_mat

    @property
    def cov_mat_inverse(self):
        if self._cov_mat_inverse is None:
            self._cov_mat_inverse = np.linalg.inv(self._cov_mat)
        return self._cov_mat_inverse

    def cost(self, parameter_values):
        _selected_par_values = np.asarray(parameter_values)[self._indices]
        _res = _selected_par_values - self._values
        return _res.dot(self.cov_mat_inverse).dot(_res)

