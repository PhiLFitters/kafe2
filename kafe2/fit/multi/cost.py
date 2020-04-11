from funcsigs import Signature, Parameter
import numpy as np

from .._base import CostFunctionBase, CostFunctionBase_Chi2

__all__ = ['MultiCostFunction', 'SharedChi2CostFunction']


class MultiCostFunction(CostFunctionBase):

    def __init__(self, singular_cost_functions):
        self._cost_function_handle = MultiCostFunction.cost_sum
        self._cost_function_argcount = len(singular_cost_functions)
        self._cost_function_signature = Signature(
            parameters=[Parameter(name='cost%s' % _i, kind=Parameter.POSITIONAL_OR_KEYWORD)
                        for _i in range(self._cost_function_argcount)]
        )
        self._assign_function_formatter()

        self._flags = {}
        self._ndf = None
        self._needs_errors = False
        for _singular_cost_function in singular_cost_functions:
            if _singular_cost_function.needs_errors:
                self._needs_errors = True
                break

    @staticmethod
    def cost_sum(*single_costs):
        return np.sum(single_costs)


class SharedChi2CostFunction(CostFunctionBase_Chi2):
    def __init__(self, errors_to_use='covariance', fallback_on_singular=True):
        CostFunctionBase_Chi2.__init__(self, errors_to_use=errors_to_use, fallback_on_singular=fallback_on_singular)
