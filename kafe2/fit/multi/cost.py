from funcsigs import Signature, Parameter
import numpy as np

from .._base import CostFunction, CostFunction_Chi2

__all__ = ['SharedCostFunction', 'MultiCostFunction']


class SharedCostFunction(CostFunction_Chi2):
    def __init__(self, fallback_on_singular=True):
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        self._COV_MAT_INVERSE_NAME = "total_cov_mat_inverse"
        super(SharedCostFunction, self).__init__(
            errors_to_use="covariance", fallback_on_singular=fallback_on_singular)


class MultiCostFunction(CostFunction):

    def __init__(self, singular_cost_functions, cost_function_names):
        super(MultiCostFunction, self).__init__(
            cost_function=MultiCostFunction.cost_sum,
            arg_names=cost_function_names,
            add_constraint_cost=True
        )
        self._needs_errors = np.any([_scf.needs_errors for _scf in singular_cost_functions])
        self._saturated = np.all([_scf.saturated for _scf in singular_cost_functions])
        self._is_chi2 = np.all([_scf.is_chi2 for _scf in singular_cost_functions])

    @staticmethod
    def cost_sum(*single_costs):
        return np.sum(single_costs)
