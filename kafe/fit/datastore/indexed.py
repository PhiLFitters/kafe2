import abc
import numpy as np

from . import DataContainerBase, DataContainerException

from ...core.error import SimpleGaussianError, MatrixGaussianError


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
            _axis = _err_dict['axis']
            _err_dict['err'].reference = self._idx_data
        self._total_error = None

    @property
    def err(self):
        _total_error = self.get_total_error()
        return _total_error.error

    # -- public methods

    def add_simple_error(self, err_val, correlation=0, relative=False):
        _err = SimpleGaussianError(err_val=err_val, corr_coeff=correlation,
                                   relative=relative, reference=self._idx_data)
        # TODO: reason not to use id() here?
        _id = id(_err)
        assert _id not in self._error_dicts
        _new_err_dict = dict(err=_err, enabled=True)
        self._error_dicts[_id] = _new_err_dict
        self._total_error = None
        return _id

    def add_matrix_error(self, axis, err_matrix, matrix_type, err_val=None, relative=False):
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


class IndexedParametricModel(IndexedContainer):
    def __init__(self):
        raise NotImplementedError