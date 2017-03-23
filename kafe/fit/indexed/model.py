import numpy as np

from scipy.misc import derivative

from .._base import ParametricModelBaseMixin, ModelFunctionBase, ModelFunctionException
from .container import IndexedContainer, IndexedContainerException
from .format import IndexedModelFunctionFormatter

class IndexedModelFunctionException(ModelFunctionException):
    pass

class IndexedModelFunction(ModelFunctionBase):
    EXCEPTION_TYPE = IndexedModelFunctionException
    FORMATTER_TYPE = IndexedModelFunctionFormatter

    def __init__(self, model_function):
        self._index_name = 'i'
        super(IndexedModelFunction, self).__init__(model_function=model_function)

    def _validate_model_function_raise(self):
        # require 'indexed' model functions to have at least one argument
        if self.argcount < 1:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' needs at least one parameter!!"
                % (self.func,))

        super(IndexedModelFunction, self)._validate_model_function_raise()

    @property
    def index_name(self):
        return self._index_name


class IndexedParametricModelException(IndexedContainerException):
    pass


class IndexedParametricModel(ParametricModelBaseMixin, IndexedContainer):
    def __init__(self, model_func, model_parameters, shape_like=None):
        # print "IndexedParametricModel.__init__(model_func=%r, model_parameters=%r)" % (model_func, model_parameters)
        if shape_like is not None:
            _data = np.zeros_like(shape_like)
            _data[:] = model_func(*model_parameters)
        else:
            _data = model_func(*model_parameters)
        super(IndexedParametricModel, self).__init__(model_func, model_parameters, _data)

    # -- private methods

    def _recalculate(self):
        # use parent class setter for 'data'
        IndexedContainer.data.fset(self, self.eval_model_function())
        self._pm_calculation_stale = False


    # -- public properties

    @property
    def data(self):
        if self._pm_calculation_stale:
            self._recalculate()
        return super(IndexedParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise IndexedParametricModelException("Parametric model data cannot be set!")

    @property
    def data_range(self):
        _data = self.data
        return np.min(_data), np.max(_data)

    # -- public methods

    def eval_model_function(self, model_parameters=None):
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        return self._model_function_handle(*_pars)

    def eval_model_function_derivative_by_parameters(self, model_parameters=None, par_dx=None):
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _pars = np.asarray(_pars)
        _par_dxs = par_dx if par_dx is not None else 1e-2 * (np.abs(_pars) + 1.0 / (1.0 + np.abs(_pars)))

        _ret = []
        for _par_idx, (_par_val, _par_dx) in enumerate(zip(_pars, _par_dxs)):
            def _chipped_func(par):
                _chipped_pars = _pars.copy()
                _chipped_pars[_par_idx] = par
                return self._model_function_handle(*_chipped_pars)

            _der_val = derivative(_chipped_func, _par_val, dx=_par_dx)
            _ret.append(_der_val)
        return np.array(_ret)