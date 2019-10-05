import inspect
import numpy as np

from .._base import CostFunctionBase


class MultiCostFunction(CostFunctionBase):

    def __init__(self, singular_cost_functions, fit_namespaces):
        self._cost_function_handle = MultiCostFunction.cost_sum
        _args = ['%scost' % _fit_namespace for _fit_namespace in fit_namespaces]
        _varargs = None
        _keywords = None
        _defaults = np.ones_like(_args)
        self._cost_function_argspec = inspect.ArgSpec(
            args=_args, varargs=_varargs, keywords=_keywords, defaults=_defaults)
        self._cost_function_argcount = len(singular_cost_functions)
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
