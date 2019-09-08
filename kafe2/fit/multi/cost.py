import numpy as np

from .._base import CostFunctionBase


class MultiCostFunction(CostFunctionBase):

    def __init__(self):
        self._cost_function_handle = MultiCostFunction.cost_sum

    @staticmethod
    def cost_sum(*single_costs):
        return np.sum(single_costs)
