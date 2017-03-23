import abc
import inspect
import numpy as np
import re
import string


class CostFunctionException(Exception):
    pass


class CostFunctionBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    EXCEPTION_TYPE = CostFunctionException

    def __init__(self, cost_function):
        self._cost_function_handle = cost_function
        self._validate_cost_function_raise()

    def _validate_cost_function_raise(self):
        self._cost_func_argspec = inspect.getargspec(self._cost_function_handle)
        if 'cost' in self._cost_func_argspec:
            raise self.__class__.EXCEPTION_TYPE(
                "The alias 'cost' for the cost function value cannot be used as an argument to the cost function!")

        if self._cost_func_argspec.varargs and self._cost_func_argspec.keywords:
            raise self.__class__.EXCEPTION_TYPE("Cost function with variable arguments (*%s, **%s) is not supported"
                                 % (self._cost_func_argspec.varargs,
                                    self._cost_func_argspec.keywords))
        elif self._cost_func_argspec.varargs:
            raise self.__class__.EXCEPTION_TYPE(
                "Cost function with variable arguments (*%s) is not supported"
                % (self._cost_func_argspec.varargs,))
        elif self._cost_func_argspec.keywords:
            raise self.__class__.EXCEPTION_TYPE(
                "Cost function with variable arguments (**%s) is not supported"
                % (self._cost_func_argspec.keywords,))
        # TODO: fail if cost function does not depend on data or model

    def __call__(self, *args, **kwargs):
        self._cost_function_handle(*args, **kwargs)

    @property
    def name(self):
        return self._cost_function_handle.__name__

    @property
    def func(self):
        return self._cost_function_handle
