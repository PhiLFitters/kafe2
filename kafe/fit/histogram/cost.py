import numpy as np

from .._base import CostFunctionBase, CostFunctionException


class HistCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to histograms.
        The function handle must be provided by the user.
        """
        super(HistCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class HistCostFunction_Chi2_NoErrors(CostFunctionBase):
    def __init__(self):
        """
        Built-in least-squares cost function calculated from data and model values,
        without considering uncertainties.
        """
        super(HistCostFunction_Chi2_NoErrors, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(data, model):
        return np.sum((data - model)**2)