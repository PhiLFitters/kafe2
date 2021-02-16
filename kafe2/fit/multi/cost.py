import numpy as np

from .._base import CostFunction, CostFunction_Chi2

__all__ = ['SharedCostFunction', 'MultiCostFunction']


class SharedCostFunction(CostFunction_Chi2):
    def __init__(self, fallback_on_singular=True):
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        self._COV_MAT_CHOLESKY_NAME = "total_cov_mat_cholesky"
        super(SharedCostFunction, self).__init__(
            errors_to_use="covariance", fallback_on_singular=fallback_on_singular,
            add_constraint_cost=False, add_determinant_cost=True)


class MultiCostFunction(CostFunction):

    def __init__(self, singular_cost_functions, cost_function_names):
        super(MultiCostFunction, self).__init__(
            cost_function=MultiCostFunction.cost_sum,
            arg_names=cost_function_names,
            add_constraint_cost=True,
            add_determinant_cost=False
        )
        self._needs_errors = np.any([_scf.needs_errors for _scf in singular_cost_functions])
        self._saturated = np.all([_scf.saturated for _scf in singular_cost_functions])
        self._is_chi2 = np.all([_scf.is_chi2 for _scf in singular_cost_functions])
        if self.is_chi2:
            self.formatter.name = "chi2"
            self.formatter.latex_name = "\\chi^2"
        else:
            self.formatter.name = "cost"
            self.formatter.latex_name = None

    @staticmethod
    def cost_sum(*single_costs):
        return np.sum(single_costs)
