import numpy as np

from ...core.error import MatrixGaussianError, SimpleGaussianError
from .._base import DataContainerException, DataContainerBase


__all__ = ['IndexedContainer', 'IndexedContainerException']


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
        super(IndexedContainer, self).__init__()
        self._data = np.array(data, dtype=dtype)

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

    def _clear_total_error_cache(self):
        self._total_error = None

    # -- public properties

    @property
    def size(self):
        """number of data points"""
        return len(self._data)

    @property
    def data(self):
        """container data (one-dimensional :py:obj:`numpy.ndarray`)"""
        return self._data.copy()  # copy to ensure no modification by user

    @data.setter
    def data(self, data):
        _data = np.squeeze(np.array(data, dtype=float))
        if len(_data.shape) > 1:
            raise IndexedContainerException("IndexedContainer data must be 1-d array of floats! Got shape: %r..." % (_data.shape,))
        self._data[:] = _data
        # reset member error references to the new data values
        for _err_dict in self._error_dicts.values():
            _err_dict['err'].reference = self._data
        self._clear_total_error_cache()

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

    @property
    def cor_mat(self):
        """absolute data correlation matrix (:py:obj:`numpy.matrix`)"""
        _total_error = self.get_total_error()
        return _total_error.cor_mat

    @property
    def data_range(self):
        """
        :return: the minimum and maximum value of the data
        """
        return np.amin(self.data), np.amax(self.data)


    # -- public methods

    def add_error(self, err_val,
                  name=None, correlation=0, relative=False):
        """
        Add an uncertainty source to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :return: error name
        :rtype: str
        """
        return super(IndexedContainer, self).add_error(
            err_val=err_val,
            name=name,
            correlation=correlation,
            relative=relative,
            reference=lambda: self._data  # set the reference appropriately
        )

    def add_matrix_error(self, err_matrix, matrix_type,
                         name=None, err_val=None, relative=False):
        """
        Add a matrix uncertainty source to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted
                         as a *relative* uncertainty
        :type relative: bool
        :return: error name
        :rtype: str
        """
        return super(IndexedContainer, self).add_matrix_error(
            err_matrix=err_matrix,
            matrix_type=matrix_type,
            name=name,
            err_val=err_val,
            relative=relative,
            reference=lambda: self._data  # set the reference appropriately
        )
