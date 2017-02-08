import abc
import inspect


class FitException(Exception):
    pass


class FitBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    CONTAINER_TYPE = None
    MODEL_TYPE = None
    EXCEPTION_TYPE = FitException
    RESERVED_NODE_NAMES = None

    # -- private methods

    def _new_data_container(self, *args, **kwargs):
        return self.__class__.CONTAINER_TYPE(*args, **kwargs)

    def _new_parametric_model(self, *args, **kwargs):
        return self.__class__.MODEL_TYPE(*args, **kwargs)

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

    def _validate_model_function_raise(self):
        self._model_func_argspec = inspect.getargspec(self._model_func_handle)
        if self._model_func_argspec.varargs and self._model_func_argspec.keywords:
            raise self.__class__.EXCEPTION_TYPE("Model function with variable arguments (*%s, **%s) is not supported"
                                      % (self._model_func_argspec.varargs,
                                         self._model_func_argspec.keywords))
        elif self._model_func_argspec.varargs:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function with variable arguments (*%s) is not supported"
                % (self._model_func_argspec.varargs,))
        elif self._model_func_argspec.keywords:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function with variable arguments (**%s) is not supported"
                % (self._model_func_argspec.keywords,))

        # check for reserved keywords
        if not self.RESERVED_NODE_NAMES.isdisjoint(set(self._model_func_argspec.args)):
            _invalid_args = self.RESERVED_NODE_NAMES.intersection(set(self._model_func_argspec.args))
            raise self.__class__.EXCEPTION_TYPE(
                "The following names are reserved and cannot be used as model function arguments: %r"
                % (_invalid_args,))

        # check for reserved keywords
        if not self.RESERVED_NODE_NAMES.isdisjoint(set(self._model_func_argspec.args)):
            _invalid_args = self.RESERVED_NODE_NAMES.intersection(set(self._model_func_argspec.args))
            raise self.__class__.EXCEPTION_TYPE(
                "The following names are reserved and cannot be used as model function arguments: %r"
                % (_invalid_args,))

    # -- public properties

    # @abc.abstractproperty
    # def data(self): pass
    #
    # @abc.abstractproperty
    # def model(self): pass
    #
    # @abc.abstractproperty
    # def data_error(self): pass
    #
    # @abc.abstractproperty
    # def data_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def data_cov_mat_inverse(self): pass
    #
    # @abc.abstractproperty
    # def model_error(self): pass
    #
    # @abc.abstractproperty
    # def model_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def model_cov_mat_inverse(self): pass
    #
    # @abc.abstractproperty
    # def total_error(self): pass
    #
    # @abc.abstractproperty
    # def total_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def total_cov_mat_inverse(self): pass
    #
    # @abc.abstractproperty
    # def parameter_values(self): pass
    #
    # @abc.abstractproperty
    # def parameter_name_value_dict(self): pass
    #
    # @abc.abstractproperty
    # def cost_function_value(self): pass

    @property
    def data_size(self):
        return self._data_container.size

    @property
    def has_model_errors(self):
        return self._param_model.has_errors

    @property
    def has_data_errors(self):
        return self._data_container.has_errors

    @property
    def has_errors(self):
        return True if self.has_data_errors or self.has_model_errors else False

    # -- public methods

    @abc.abstractmethod
    def add_simple_error(self): pass

    @abc.abstractmethod
    def add_matrix_error(self): pass

    @abc.abstractmethod
    def do_fit(self): pass