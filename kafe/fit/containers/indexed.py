import abc
import numpy as np

from . import DataContainerBase, DataContainerException, ParametricModelBaseMixin

from ...core.error import SimpleGaussianError, MatrixGaussianError

from functools import partial
from scipy.misc import derivative

class IndexedContainerException(DataContainerException):
    pass


class IndexedContainer(DataContainerBase):
    def __init__(self, data, dtype=float):
        self._idx_data = np.array(data, dtype=dtype)
        self._error_dicts = {}
        self._total_error = None

    # -- private methods

    def _calculate_total_error(self):
        _sz = self.size
        _tmp_cov_mat = np.zeros((_sz, _sz))
        for _err_dict in self._error_dicts.values():
            if not _err_dict['enabled']:
                continue
            _tmp_cov_mat += _err_dict['err'].cov_mat

        _total_err = MatrixGaussianError(_tmp_cov_mat, 'cov', relative=False, reference=self.data)
        self._total_error = _total_err

    # -- public properties

    @property
    def size(self):
        return len(self._idx_data)

    @property
    def data(self):
        return self._idx_data.copy()  # copy to ensure no modification by user

    @data.setter
    def data(self, data):
        _data = np.squeeze(np.array(data, dtype=float))
        if len(_data.shape) > 1:
            raise IndexedContainerException("IndexedContainer data must be 1-d array of floats! Got shape: %r..." % (_data.shape,))
        self._idx_data[:] = _data
        # reset member error references to the new data values
        for _err_dict in self._error_dicts.values():
            _err_dict['err'].reference = self._idx_data
        self._total_error = None

    @property
    def err(self):
        _total_error = self.get_total_error()
        return _total_error.error

    @property
    def cov_mat(self):
        _total_error = self.get_total_error()
        return _total_error.cov_mat

    @property
    def cov_mat_inverse(self):
        _total_error = self.get_total_error()
        return _total_error.cov_mat_inverse

    # -- public methods

    def add_simple_error(self, err_val, correlation=0, relative=False):
        try:
            err_val.ndim
        except AttributeError:
            err_val = np.ones(self.size) * err_val
        _err = SimpleGaussianError(err_val=err_val, corr_coeff=correlation,
                                   relative=relative, reference=self._idx_data)
        # TODO: reason not to use id() here?
        _id = id(_err)
        assert _id not in self._error_dicts
        _new_err_dict = dict(err=_err, enabled=True)
        self._error_dicts[_id] = _new_err_dict
        self._total_error = None
        return _id

    def add_matrix_error(self, err_matrix, matrix_type, err_val=None, relative=False):
        _err = MatrixGaussianError(err_matrix=err_matrix, matrix_type=matrix_type, err_val=err_val,
                                   relative=relative, reference=self._idx_data)
        # TODO: reason not to use id() here?
        _id = id(_err)
        assert _id not in self._error_dicts
        _new_err_dict = dict(err=_err, enabled=True)
        self._error_dicts[_id] = _new_err_dict
        self._total_error = None
        return _id

    def disable_error(self, err_id):
        _err_dict = self._error_dicts.get(err_id, None)
        if _err_dict is None:
            raise IndexedContainerException("No error with id %d!" % (err_id,))
        _err_dict['enabled'] = False
        self._total_error = None

    def get_total_error(self):
        if self._total_error is None:
            self._calculate_total_error()
        return self._total_error


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