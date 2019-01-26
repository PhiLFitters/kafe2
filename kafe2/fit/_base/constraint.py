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
    # TODO check input valid
    def __init__(self, par_index, par_mean, par_uncertainty):
        self._par_index = par_index
        self._par_mean = par_mean
        self._par_uncertainty = par_uncertainty
        super(GaussianSimpleParameterConstraint).__init__()

    def cost(self, parameter_values):
        return ((parameter_values[self._par_index] - self._par_mean) / self._par_uncertainty) ** 2


class GaussianMatrixParameterConstraint(ParameterConstraint):
    # TODO documentation
    # TODO check input valid
    def __init__(self, par_indices, par_means, par_cov_mat):
        self._par_indices = np.array(par_indices)
        self._par_means = np.array(par_means)
        self._par_cov_mat = np.array(par_cov_mat)
        self._par_cov_mat_inverse = None
        super(GaussianMatrixParameterConstraint).__init__()

    @property
    def par_cov_mat_inverse(self):
        if self._par_cov_mat_inverse is None:
            self._par_cov_mat_inverse = np.linalg.inv(self._par_cov_mat)
        return self._par_cov_mat_inverse

    def cost(self, parameter_values):
        _selected_par_values = np.asarray(parameter_values)[self._par_indices]
        _res = _selected_par_values - self._par_means
        return _res.dot(self.par_cov_mat_inverse).dot(_res)

