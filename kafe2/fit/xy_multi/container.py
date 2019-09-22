import numpy as np
import six

from ...core.error import MatrixGaussianError, SimpleGaussianError
from ..indexed import IndexedContainer
from ..indexed.container import IndexedContainerException


__all__ = ["XYMultiContainer"]


class XYMultiContainerException(IndexedContainerException):
    pass


class XYMultiContainer(IndexedContainer):
    """
    This object is a specialized data container for *xy* data associated with 1 or more datasets.

    """
    _AXIS_SPEC_DICT = {0:0, 1:1, '0':0, '1':1, 'x':0, 'y':1}

    def __init__(self, xy_data, dtype=float):
        """
        Construct a container for *xy* data:

        :param xy_data: arrays containing measurement *x* values and measurement *y* values
        :type xy_data: 2xn iterable of type <dtype>, or an iterable thereof, n can vary
        :param dtype: data type of the measurements
        :type dtype: type
        """
        
        try:
            xy_data[0][0] #raises IndexError if xy_data has < 2 dimensions
        except (IndexError, TypeError):
            raise XYMultiContainerException("xy_data has to have at least 2 dimensions!")
        try:
            xy_data[0][0][0] #raises IndexError if xy_data has 2 dimensions
        except (IndexError, TypeError):
            xy_data = [xy_data]
        _xy_datasets = []
        _total_length = 0
        self._data_indices = [0]
        for _index, _xy_dataset in enumerate(xy_data):
            _xy_dataset = np.array(_xy_dataset, dtype=dtype)
            if _xy_dataset.ndim != 2 or _xy_dataset.shape[0] != 2:
                raise XYMultiContainerException("xy dataset with index %s cannot be converted into a 2xn numpy array!" % _index)
            _xy_datasets.append(_xy_dataset)
            _total_length += _xy_dataset.shape[1]
            self._data_indices.append(_total_length)
        self._xy_data = np.empty((2, _total_length), dtype=dtype)
        for _index, _xy_dataset in enumerate(_xy_datasets):
            _upper = self._data_indices[_index + 1]
            _lower = self._data_indices[_index]
            self._xy_data[0][_lower : _upper] = _xy_dataset[0]
            self._xy_data[1][_lower : _upper] = _xy_dataset[1]

        self._error_dicts = {}
        self._xy_total_errors = None


    # -- private methods

    @staticmethod
    def _find_axis_raise(axis_spec):
        try:
            axis_spec = axis_spec.lower()
        except AttributeError:
            # integers have no .lower() method
            pass
        _axis_id = XYMultiContainer._AXIS_SPEC_DICT.get(axis_spec, None)
        if _axis_id is None:
            raise XYMultiContainerException("No axis with id %r!" % (axis_spec,))
        return _axis_id

    def _get_data_for_axis(self, axis_id):
        return self._xy_data[axis_id]

    def _calculate_total_error(self):
        _sz = self.size
        _tmp_cov_mat_x = np.zeros((_sz, _sz))
        _tmp_cov_mat_y = np.zeros((_sz, _sz))
        for _err_dict in self._error_dicts.values():
            if not _err_dict['enabled']:
                continue
            assert _err_dict['axis'] in (0, 1)
            if _err_dict['axis'] == 0:
                _tmp_cov_mat_x += _err_dict['err'].cov_mat
            elif _err_dict['axis'] == 1:
                _tmp_cov_mat_y += _err_dict['err'].cov_mat

        _total_err_x = MatrixGaussianError(_tmp_cov_mat_x, 'cov', relative=False, reference=self.x)
        _total_err_y = MatrixGaussianError(_tmp_cov_mat_y, 'cov', relative=False, reference=self.y)
        self._xy_total_errors = [_total_err_x, _total_err_y]

    def _clear_total_error_cache(self):
        """recalculate total errors next time they are needed"""
        self._xy_total_errors = None

    def _calculate_uncor_error_cov_mat(self, axis):
        #calculate y uncorrelated covariance matrix
        _sz = self.size
        _tmp_uncor_cov_mat = np.zeros((_sz, _sz))
        for _err_dict in self._error_dicts.values():
            if not _err_dict['enabled']:
                continue
            if not _err_dict['axis'] == axis:
                continue
            _err = _err_dict["err"]
            if isinstance(_err, MatrixGaussianError):
                _tmp_uncor_cov_mat+=_err.cov_mat
            else:
                _tmp_uncor_cov_mat+=_err.cov_mat_uncor
        return np.array(_tmp_uncor_cov_mat)

    def _calculate_y_nuisance_cor_design_matrix(self):
        """calculate the design matrix containing the correlated parts of all y uncertainties"""

        _y_cor_errors = self.get_matching_errors(
            matching_criteria=dict(
                axis=1,
                enabled=True,
                correlated=True
            )
        )

        _data_size = self.size
        _err_size = len(_y_cor_errors)
        _nuisance_ycor_design_matrix = np.zeros((_err_size, _data_size))
        for _col, (_err_name, _err) in enumerate(six.iteritems(_y_cor_errors)):
            _nuisance_ycor_design_matrix[_col, :] = _err.error_cor

        return np.array(_nuisance_ycor_design_matrix)

    # def _calculate_x_nuisance_cor_design_matrix(self):
    #     """calculate the design matrix containing the correlated parts of all x uncertainties"""
    #
    #     _x_cor_errors = self.get_matching_errors(
    #         matching_criteria=dict(
    #             axis=0,
    #             enabled=True,
    #             correlated=True
    #         )
    #     )
    #
    #     _data_size = self.size
    #     _err_size = len(_x_cor_errors)
    #     _nuisance_xcor_design_matrix = np.zeros((_err_size, _data_size))
    #     for _col, (_err_name, _err) in enumerate(six.iteritems(_x_cor_errors)):
    #         _nuisance_xcor_design_matrix[_col, :] = _err.error_cor
    #
    #     return np.array(_nuisance_xcor_design_matrix)

    # -- public properties

    @property
    def size(self):
        """total number of data points"""
        return self._xy_data.shape[1]

    @property
    def data(self):
        """container data (both *x* and *y*, two-dimensional :py:obj:`numpy.ndarray`)"""
        return self._xy_data.copy()  # copy to ensure no modification by user

    @data.setter
    def data(self, new_data):
        #TODO new_data and data_indices can be inconsistent
        _new_data = np.asarray(new_data)
        if _new_data.ndim != 2:
            raise XYMultiContainerException("XYMultiContainer data must be 2-d array of floats! Got shape: %r..." % (_new_data.shape,))
        if _new_data.shape[0] == 2:
            self._xy_data = _new_data.copy()
        elif _new_data.shape[1] == 2:
            self._xy_data = _new_data.T.copy()
        else:
            raise XYMultiContainerException(
                "XYMultiContainer data length must be 2 in at least one axis! Got shape: %r..." % (_new_data.shape,))
        self._clear_total_error_cache()

    @property
    def x(self):
        """container *x* data (one-dimensional :py:obj:`numpy.ndarray`)"""
        return self._get_data_for_axis(0)

    @x.setter
    def x(self, new_x):
        _new_x_data = np.squeeze(np.array(new_x))
        if len(_new_x_data.shape) > 1:
            raise XYMultiContainerException("XYMultiContainer 'x' data must be 1-d array of floats! Got shape: %r..." % (_new_x_data.shape,))
        self._xy_data[0,:] = new_x
        for _err_dict in self._error_dicts.values():
            if _err_dict['axis'] == 0:
                _err_dict['err'].reference = self._get_data_for_axis(0)
        self._clear_total_error_cache()

    @property
    def x_err(self):
        """absolute total data *x*-uncertainties (one-dimensional :py:obj:`numpy.ndarray`)"""
        _total_error_x = self.get_total_error(axis=0)
        return _total_error_x.error

    @property
    def x_cov_mat(self):
        """absolute data *x* covariance matrix (:py:obj:`numpy.matrix`)"""
        _total_error_x = self.get_total_error(axis=0)
        return _total_error_x.cov_mat

    @property
    def x_cov_mat_inverse(self):
        """inverse of absolute data *x* covariance matrix (:py:obj:`numpy.matrix`), or ``None`` if singular"""
        _total_error_x = self.get_total_error(axis=0)
        return _total_error_x.cov_mat_inverse

    @property
    def x_cor_mat(self):
        """absolute data *x* correlation matrix (:py:obj:`numpy.matrix`)"""
        _total_error_x = self.get_total_error(axis=0)
        return _total_error_x.cor_mat

    @property
    def y(self):
        return self._get_data_for_axis(1)

    @y.setter
    def y(self, new_y):
        """container *y* data (one-dimensional :py:obj:`numpy.ndarray`)"""
        _new_y_data = np.squeeze(np.array(new_y))
        if len(_new_y_data.shape) > 1:
            raise XYMultiContainerException("XYMultiContainer 'y' data must be 1-d array of floats! Got shape: %r..." % (_new_y_data.shape,))
        self._xy_data[1,:] = new_y
        for _err_dict in self._error_dicts.values():
            if _err_dict['axis'] == 1:
                _err_dict['err'].reference = self._get_data_for_axis(1)
        self._clear_total_error_cache()

    @property
    def y_err(self):
        """absolute total data *y*-uncertainties (one-dimensional :py:obj:`numpy.ndarray`)"""
        _total_error_y = self.get_total_error(axis=1)
        return _total_error_y.error

    @property
    def y_cov_mat(self):
        """absolute data *y* covariance matrix (:py:obj:`numpy.matrix`)"""
        _total_error_y = self.get_total_error(axis=1)
        return _total_error_y.cov_mat

    @property
    def y_cov_mat_inverse(self):
        """inverse of absolute data *y* covariance matrix (:py:obj:`numpy.matrix`), or ``None`` if singular"""
        _total_error_y = self.get_total_error(axis=1)
        return _total_error_y.cov_mat_inverse

    @property
    def y_cor_mat(self):
        """absolute data *y* correlation matrix (:py:obj:`numpy.matrix`)"""
        _total_error_y = self.get_total_error(axis=1)
        return _total_error_y.cor_mat

    @property
    def x_range(self):
        """x data range across all datasets"""
        _x = self.x
        return np.min(_x), np.max(_x)

    def get_x_range(self, index):
        """x data range for the dataset with the specified index"""
        _x = self.x[self.data_indices[index] : self.data_indices[index + 1]]
        return np.min(_x), np.max(_x)

    @property
    def y_range(self):
        """y data range across all datasets"""
        _y = self.y
        return np.min(_y), np.max(_y)

    @property
    def y_uncor_cov_mat(self):
        # y uncorrelated covariance matrix
        _y_uncor_cov_mat = self._calculate_uncor_error_cov_mat(axis=1)
        return _y_uncor_cov_mat

    @property
    def y_uncor_cov_mat_inverse(self):
        # y uncorrelated inverse covariance matrix
        return np.linalg.inv(self.y_uncor_cov_mat)

    @property
    def _y_nuisance_cor_design_mat(self):
        """design matrix containing the correlated parts of all y uncertainties"""
        _nuisance_y_cor_cov_mat = self._calculate_y_nuisance_cor_design_matrix()
        return _nuisance_y_cor_cov_mat

    @property
    def x_uncor_cov_mat(self):
        # x uncorrelated covariance matrix
        _x_uncor_cov_mat = self._calculate_uncor_error_cov_mat(axis=0)
        return _x_uncor_cov_mat

    @property
    def x_uncor_cov_mat_inverse(self):
        # x uncorrelated inverse covariance matrix
        return np.linalg.inv(self.x_uncor_cov_mat)

    # @property TODO: correlated x-errors
    # def nuisance_x_cor_cov_mat(self):
    #      """design matrix containing the correlated parts of all x uncertainties"""
    #     _nuisance_x_cor_cov_mat = self._calculate_nuisance_x_cor_error_cov_mat()
    #     return _nuisance_x_cor_cov_mat

    # -- public methods

    def add_simple_error(self, axis, err_val, name=None, model_index=None, correlation=0, relative=False):
        """
        Add a simple uncertainty source for an axis to the data container.
        Returns an error name which uniquely identifies the created error source.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param model_index: the index of the dataset/model with which the error will be associated (only relevant for
                    the error dictionary)
        :type model_index: int
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :return: error name
        :rtype: str
        """
        #TODO update documentation
        _axis = self._find_axis_raise(axis)
        try:
            err_val.ndim   # will raise if simple float
        except AttributeError:
            err_val = np.asarray(err_val, dtype=float)

        
        if model_index is None:
            # if dimensionless numpy array (i.e. float64), add a dimension
            _err_all_datasets = np.ones(self.size) * err_val if err_val.ndim == 0 else err_val
        else:
            if model_index < 0 or model_index >= self.num_datasets:
                raise XYMultiContainerException("There is no corresponding dataset for index %s" % model_index)
            _lower, _upper = self.get_data_bounds(model_index)
            if err_val.ndim == 0:  # if dimensionless numpy array (i.e. float64), add a dimension
                _partial_err = np.ones(_upper - _lower) * err_val
            elif err_val.size == _upper - _lower:
                _partial_err = err_val
            else:
                raise XYMultiContainerException("Dataset %s has %s data points but err_val has size %s" % 
                                                (model_index, _upper - _lower, err_val.size))
            #Error for all datasets 0 except for the one specified
            _err_all_datasets = np.zeros(self.size)
            _err_all_datasets[_lower:_upper] = _partial_err
                
        _err = SimpleGaussianError(err_val=_err_all_datasets, corr_coeff=correlation,
                                   relative=relative, reference=self._get_data_for_axis(_axis))
        _name = self._add_error_object(name=name, model_index=model_index, error_object=_err, axis=_axis)
        return _name

    def add_matrix_error(self, axis, err_matrix, matrix_type, name=None, model_index=None, err_val=None, relative=False):
        """
        Add a matrix uncertainty source for an axis and for one specific dataset to the data container.
        Returns an error name which uniquely identifies the created error source.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :return: error name
        :rtype: str
        """
        _axis = self._find_axis_raise(axis)
        _err = MatrixGaussianError(
            err_matrix=err_matrix, matrix_type=matrix_type, err_val=err_val,
            relative=relative, reference=self._get_data_for_axis(_axis)
        )
        _name = self._add_error_object(name=name, error_object=_err, axis=_axis)
        return _name

    def get_total_error(self, axis):
        """
        Get the error object representing the total uncertainty for an axis.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int

        :return: error object representing the total uncertainty
        :rtype: :py:class:`~kafe2.core.error.MatrixGaussianError`
        """
        _axis = self._find_axis_raise(axis)
        if self._xy_total_errors is None:
            self._calculate_total_error()
        return self._xy_total_errors[_axis]

    @property
    def has_x_errors(self):
        """``True`` if at least one *x* uncertainty source is defined for the data container"""
        for _err_dict in self._error_dicts.values():
            if _err_dict['axis'] == 0:
                return True
        return False

    @property
    def has_uncor_x_errors(self):
        """``True`` if at least one *x* uncertainty source, which is not fully correlated, is defined for the data container"""
        for _err_dict in self._error_dicts.values():
            if _err_dict['axis'] == 0 and _err_dict['err'].corr_coeff != 1.0:
                return True
        return False

    @property
    def has_y_errors(self):
        """``True`` if at least one *x* uncertainty source is defined for the data container"""
        for _err_dict in self._error_dicts.values():
            if _err_dict['axis'] == 1:
                return True
        return False


    @property
    def data_indices(self):
        """the indices at which the *xy* datasets start/stop, the nth dataset starts at the nth data index and stops at the
        (n + 1)th data index"""
        return self._data_indices
    
    @property
    def num_datasets(self):
        """the number of datasets in this container"""
        #call property instead of self._data_indices because models override it
        return len(self.data_indices) - 1
    
    def get_data_bounds(self, index):
        """the bounds of the dataset with the specified index"""
        return self.data_indices[index], self.data_indices[index + 1]
    
    def get_splice(self, data, index):
        """utility function that splices the given data according to the dataset bounds of the specified index"""
        return data[self.data_indices[index] : self.data_indices[index + 1]]
    
    @property
    def all_datasets_same_size(self):
        """True if all datasets contain the same number of *xy* points"""
        _lower, _upper = self.get_data_bounds(0)
        _diff = _upper - _lower
        for _i in range(1, self.num_datasets):
            _lower, _upper = self.get_data_bounds(0)
            if _upper - _lower != _diff:
                return False
        return True
