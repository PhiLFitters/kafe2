import abc
import inspect

from .format import ModelParameterFormatter, ModelFunctionFormatter


__all__ = ["ParametricModelBaseMixin", "ModelFunctionBase", "ModelFunctionException"]


class ParametricModelBaseMixin(object):
    """
    A "mixin" class for representing a parametric model.
    Inheriting from this class in addition to a data container class
    additionally stores a Python function handle referring to the
    model function. The argument structure of this function must
    be compatible with the data container type and it must return
    a numpy array of the same shape as the
    :py:meth:`~kafe.fit._base.DataContainerBase.data` property of
    the data container.

    This mixin class introduces an additional :py:func:`parameters` property for
    the object, which can be used to obtain and set the values of the parameter

    Derived classes should inherit from :py:class:`ParametricModelBaseMixin` and the
    relevant data container (in that order).
    """
    def __init__(self, model_func, model_parameters, *args, **kwargs):
        """
        Mixin constructor: sets and initialized the model function.

        :param model_func: handle of Python function (the model function)
        :param model_parameters: iterable of parameter values with which the model function should be initialized
        """
        self._model_function_handle = model_func
        self.parameters = model_parameters
        super(ParametricModelBaseMixin, self).__init__(*args, **kwargs)

    @property
    def parameters(self):
        """Model parameter values"""
        return self._model_parameters

    @parameters.setter
    def parameters(self, parameters):
        """Setter for parameter values"""
        self._model_parameters = parameters

        # flag: recalculate the model values next time they are requested
        self._pm_calculation_stale = True
        self._clear_total_error_cache()


class ModelFunctionException(Exception):
    pass


class ModelFunctionBase(object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    model functions.

    In order to be used as a model function, a native Python function must be wrapped
    by an object whose class derives from this base class.
    There is a dedicated :py:class:`ModelFunction` specialization for each type of
    data container.

    This class provides the basic functionality used by all :py:class:`ModelFunction` objects.
    These use introspection (:py:mod:`inspect`) for determining the parameter structure of the
    model function and to ensure the function can be used as a model function (validation).

    """
    __metaclass__ = abc.ABCMeta

    EXCEPTION_TYPE = ModelFunctionException
    FORMATTER_TYPE = ModelFunctionFormatter

    def __init__(self, model_function):
        """
        Construct :py:class:`ModelFunction` object (a wrapper for a native Python function):

        :param model_function: function handle
        """
        self._model_function_handle = model_function
        self._model_function_argspec = inspect.getargspec(self._model_function_handle)
        self._model_function_argcount = self._model_function_handle.__code__.co_argcount
        self._validate_model_function_raise()
        self._assign_function_formatter()

    def _validate_model_function_raise(self):
        if self._model_function_argspec.varargs and self._model_function_argspec.keywords:
            raise self.__class__.EXCEPTION_TYPE("Model function with variable arguments (*%s, **%s) is not supported"
                                                % (self._model_function_argspec.varargs,
                                                   self._model_function_argspec.keywords))
        elif self._model_function_argspec.varargs:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function with variable arguments (*%s) is not supported"
                % (self._model_function_argspec.varargs,))
        elif self._model_function_argspec.keywords:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function with variable arguments (**%s) is not supported"
                % (self._model_function_argspec.keywords,))

    def _get_parameter_formatters(self):
        return [ModelParameterFormatter(name=_pn, value=_pv, error=None)
                for _pn, _pv in zip(self.argspec.args, self.argvals)]

    def _assign_function_formatter(self):
        self._formatter = self.__class__.FORMATTER_TYPE(
            self.name, arg_formatters=self._get_parameter_formatters())

    def __call__(self, *args, **kwargs):
        self._model_function_handle(*args, **kwargs)

    @property
    def name(self):
        """The model function name (a valid Python identifier)"""
        return self._model_function_handle.__name__

    @property
    def func(self):
        """The model function handle"""
        return self._model_function_handle

    @property
    def argspec(self):
        """The model function argument specification, as returned by :py:meth:`inspect.getargspec`"""
        return self._model_function_argspec

    @property
    def argcount(self):
        """The number of arguments the model function accepts
        (including any independent variables which are not parameters)"""
        return self._model_function_argcount

    @property
    def argvals(self):
        """The current values of the function arguments (**not yet implemented**, returns an array of zeros)"""
        # TODO: decide whether to store these (that's actually what ParametricModelMixin is for...)
        return [0.0] * (self.argcount)

    @property
    def formatter(self):
        """The :py:obj:`ModelFunctionFormatter`-derived object for this function"""
        return self._formatter

    @property
    def argument_formatters(self):
        """The :py:obj:`ModelParameterFormatter`-derived objects for the function arguments"""
        return self._formatter.arg_formatters
