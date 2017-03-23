import numpy as np

from .._base import CostFunctionBase, CostFunctionException


class XYCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to xy data.
        The function handle must be provided by the user.
        """
        super(XYCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class XYCostFunction_Chi2_NoErrors(CostFunctionBase):
    def __init__(self):
        super(XYCostFunction_Chi2_NoErrors, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(y_data, y_model):
        return np.sum((y_data - y_model)**2)