import numpy as np

from ...core.error import MatrixGaussianError, SimpleGaussianError
from .._base import DataContainerException, DataContainerBase


__all__ = ["IndexedContainer"]


class IndexedContainerException(DataContainerException):
    pass


class IndexedContainer(DataContainerBase):
    """
    This object is a specialized data container for series of indexed measurements.

    """
    def __init__(self, data, dtype=float):
        """
        Construct a container for indexed data:

        :param data: a one-dimensional array of measurements
        :type data: iterable of type <dtype>
        :param dtype: data type of the measurements
        :type dtype: type
        """
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
        """number of data points"""
        return len(self._idx_data)

    @property
    def data(self):
        """container data (one-dimensional :py:obj:`numpy.ndarray`)"""
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
        """absolute total data uncertainties (one-dimensional :py:obj:`numpy.ndarray`)"""
        _total_error = self.get_total_error()
        return _total_error.error

    @property
    def cov_mat(self):
        """absolute data covariance matrix (:py:obj:`numpy.matrix`)"""
        _total_error = self.get_total_error()
        return _total_error.cov_mat

    @property
    def cov_mat_inverse(self):
        """inverse of absolute data covariance matrix (:py:obj:`numpy.matrix`), or ``None`` if singular"""
        _total_error = self.get_total_error()
        return _total_error.cov_mat_inverse

    # -- public methods

    def add_simple_error(self, err_val, correlation=0, relative=False):
        """
        Add a simple uncertainty source to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :return: error id
        :rtype: int
        """
        try:
            err_val.ndim   # will raise if simple float
        except AttributeError:
            err_val = np.asarray(err_val, dtype=float)

        if err_val.ndim == 0:  # if dimensionless numpy array (i.e. float64), add a dimension
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
        """
        Add a matrix uncertainty source to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :return: error id
        :rtype: int
        """
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
        """
        Temporarily disable an uncertainty source so that it doesn't count towards calculating the
        total uncertainty.

        :param err_id: error id
        :type err_id: int
        """
        _err_dict = self._error_dicts.get(err_id, None)
        if _err_dict is None:
            raise IndexedContainerException("No error with id %d!" % (err_id,))
        _err_dict['enabled'] = False
        self._total_error = None

    def get_total_error(self):
        """
        Get the error object representing the total uncertainty.

        :return: error object representing the total uncertainty
        :rtype: :py:class:`~kafe.core.error.MatrixGaussianError`
        """
        if self._total_error is None:
            self._calculate_total_error()
        return self._total_error