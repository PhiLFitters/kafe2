import inspect
import numpy as np

from .._base import ParametricModelBaseMixin, ModelFunctionBase, ModelFunctionException
from .container import HistContainer, HistContainerException

class HistModelFunctionException(ModelFunctionException):
    pass

class HistModelFunction(ModelFunctionBase):
    EXCEPTION_TYPE = HistModelFunctionException
    def __init__(self, model_density_function, model_density_antiderivative=None):
        self._x_name = 'x'
        super(HistModelFunction, self).__init__(model_function=model_density_function)
        self._antiderivative = model_density_antiderivative
        self._validate_model_function_antiderivative_raise()

    def _validate_model_function_raise(self):
        # require 'hist' model function agruments to include 'x'
        if self.x_name not in self.argspec.args:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' must have independent variable '%s' among its arguments!"
                % (self.func, self.x_name))

        # require 'hist' model functions to have more than two arguments
        if self.argcount < 2:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' needs at least one parameter beside independent variable '%s'!"
                % (self.func, self.x_name))

        # evaluate general model function requirements
        super(HistModelFunction, self)._validate_model_function_raise()

    def _validate_model_function_antiderivative_raise(self):
        if self.antiderivative is None:
            return

        _model_func_antider_argspec = inspect.getargspec(self.antiderivative)

        # require antiderivative and density to have the same arguments
        if self.argspec.args != _model_func_antider_argspec.args:
            raise self.__class__.EXCEPTION_TYPE(
                "Model density function and its antiderivative have different argument structures:"
                "(%r vs %r)"
                % (self.argspec.args, _model_func_antider_argspec.args))


    @property
    def x_name(self):
        return self._x_name

    @property
    def antiderivative(self):
        return self._antiderivative


class HistParametricModelException(HistContainerException):
    pass


class HistParametricModel(ParametricModelBaseMixin, HistContainer):
    def __init__(self, n_bins, bin_range, model_density_func, model_parameters, bin_edges=None, model_density_func_antiderivative=None):
        # print "IndexedParametricModel.__init__(model_func=%r, model_parameters=%r)" % (model_func, model_parameters)
        self._model_density_func_antider_handle = model_density_func_antiderivative
        super(HistParametricModel, self).__init__(model_density_func, model_parameters, n_bins, bin_range,
                                                  bin_edges=bin_edges, fill_data=None, dtype=float)

    # -- private methods

    def _eval_model_func_density_integral_over_bins(self):
        _as = self._bin_edges[:-1]
        _bs = self._bin_edges[1:]
        assert len(_as) == len(_bs) == self.size
        if self._model_density_func_antider_handle is not None:
            _fval_antider_as = self._model_density_func_antider_handle(_as, *self._model_parameters)
            _fval_antider_bs = self._model_density_func_antider_handle(_bs, *self._model_parameters)
            assert len(_fval_antider_as) == len(_fval_antider_bs) == self.size
            _int_val = np.asarray(_fval_antider_bs) - np.asarray(_fval_antider_as)
        else:
            import scipy.integrate as integrate
            _integrand_func = lambda x: self._model_function_handle(x, *self._model_parameters)
            # TODO: find more efficient alternative
            _int_val = np.zeros(self.size)
            for _i, (_a, _b) in enumerate(zip(_as, _bs)):
                _int_val[_i], _ = integrate.quad(_integrand_func, _a, _b)
        return _int_val

    def _recalculate(self):
        # don't use parent class setter for 'data' -> set directly
        self._idx_data[1:-1] = self._eval_model_func_density_integral_over_bins()
        self._pm_calculation_stale = False


    # -- public properties

    @property
    def data(self):
        if self._pm_calculation_stale:
            self._recalculate()
        return super(HistParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise HistParametricModelException("Parametric model data cannot be set!")

    # -- public methods

    def fill(self, entries):
        raise HistParametricModelException("Parametric model of histogram cannot be filled!")