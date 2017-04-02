import abc
import inspect
import numpy as np
import re
import string


class CostFunctionException(Exception):
    pass


class CostFunctionBase(object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    cost functions.

    Any Python function returning a ``float`` can be used as a cost function,
    although a number of common cost functions are provided as built-ins for
    all fit types.

    In order to be used as a model function, a native Python function must be wrapped
    by an object whose class derives from this base class.
    There is a dedicated :py:class:`CostFunction` specialization for each type of
    fit.

    This class provides the basic functionality used by all :py:class:`CostFunction` objects.
    These use introspection (:py:mod:`inspect`) for determining the parameter structure of the
    cost function and to ensure the function can be used as a cost function (validation).

    """
    __metaclass__ = abc.ABCMeta

    EXCEPTION_TYPE = CostFunctionException

    def __init__(self, cost_function):
        """
        Construct :py:class:`CostFunction` object (a wrapper for a native Python function):

        :param cost_function: function handle
        """
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
        return self._cost_function_handle(*args, **kwargs)

    @property
    def name(self):
        """The cost function name (a valid Python identifier)"""
        return self._cost_function_handle.__name__

    @property
    def func(self):
        """The cost function handle"""
        return self._cost_function_handle
