import numpy as np

from scipy.misc import derivative

from .._base import ParametricModelBaseMixin
from .container import IndexedContainer, IndexedContainerException


class IndexedParametricModelException(IndexedContainerException):
    pass


class IndexedParametricModel(ParametricModelBaseMixin, IndexedContainer):
    def __init__(self, model_func, model_parameters):
        # print "IndexedParametricModel.__init__(model_func=%r, model_parameters=%r)" % (model_func, model_parameters)
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