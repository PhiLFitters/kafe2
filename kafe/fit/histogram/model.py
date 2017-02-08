import numpy as np

from .._base import ParametricModelBaseMixin
from .container import HistContainer, HistContainerException


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