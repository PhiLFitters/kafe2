import abc
import inspect

class ParametricModelBaseMixin(object):
    """
    Mixin class. Defines additional properties and methods to be 'mixed into' another class.
    """
    def __init__(self, model_func, model_parameters, *args, **kwargs):
        # print "ParametricModelBaseMixin.__init__(model_func=%r, model_parameters=%rb, args=%r, kwargs=%r)" % (model_func, model_parameters, args, kwargs)
        self._model_function_handle = model_func
        self.parameters = model_parameters
        super(ParametricModelBaseMixin, self).__init__(*args, **kwargs)

    @property
    def parameters(self):
        return self._model_parameters

    @parameters.setter
    def parameters(self, parameters):
        self._model_parameters = parameters
        self._pm_calculation_stale = True


class ModelFunctionException(Exception):
    pass


class ModelFunctionBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    EXCEPTION_TYPE = ModelFunctionException
    RESERVED_NODE_NAMES = set()

    def __init__(self, model_function):
        self._model_function_handle = model_function
        self._model_function_argspec = inspect.getargspec(self._model_function_handle)
        self._model_function_argcount = self._model_function_handle.func_code.co_argcount
        self._validate_model_function_raise()

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

    def __call__(self, *args, **kwargs):
        self._model_function_handle(*args, **kwargs)

    @property
    def name(self):
        return self._model_function_handle.__name__

    @property
    def func(self):
        return self._model_function_handle

    @property
    def argspec(self):
        return self._model_function_argspec

    @property
    def argcount(self):
        return self._model_function_argcount
