import numpy as np

from .._base import CostFunctionBase, CostFunctionException


class IndexedCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to indexed measurements.
        The function handle must be provided by the user.
        """
        super(IndexedCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class IndexedCostFunction_Chi2_NoErrors(CostFunctionBase):
    def __init__(self):
        super(IndexedCostFunction_Chi2_NoErrors, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(data, model):
        return np.sum((data - model)**2)