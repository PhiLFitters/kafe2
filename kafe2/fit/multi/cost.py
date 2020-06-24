from funcsigs import Signature, Parameter
import numpy as np

from .._base import CostFunction, CostFunction_Chi2

__all__ = ['MultiCostFunction']


class MultiCostFunction(CostFunction):

    def __init__(self, singular_cost_functions):
        super(MultiCostFunction, self).__init__(
            cost_function=MultiCostFunction.cost_sum,
            arg_names=["cost%s" % _i for _i in range(len(singular_cost_functions))]
        )
        self._needs_errors = False
        for _singular_cost_function in singular_cost_functions:
            if _singular_cost_function.needs_errors:
                self._needs_errors = True
                break

    @staticmethod
    def cost_sum(*single_costs):
        return np.sum(single_costs)
