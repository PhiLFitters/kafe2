from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ...core.error import CovMat
from ...tools import print_dict_as_table
from ...config import kc
from .._base import FitException, FitBase, DataContainerBase, ModelFunctionBase
from .container import XYContainer
from .cost import XYCostFunction_Chi2, STRING_TO_COST_FUNCTION
from .model import XYParametricModel
from .plot import XYPlotAdapter
from ..util import function_library, add_in_quadrature, collect, invert_matrix


__all__ = ['XYFit', 'XYFitException']


class XYFitException(FitException):
    pass


class XYFit(FitBase):
    CONTAINER_TYPE = XYContainer
    MODEL_TYPE = XYParametricModel
    MODEL_FUNCTION_TYPE = ModelFunctionBase
    PLOT_ADAPTER_TYPE = XYPlotAdapter
    EXCEPTION_TYPE = XYFitException
    RESERVED_NODE_NAMES = {'y_data', 'y_model', 'cost',
                           'x_error', 'y_data_error', 'y_model_error', 'total_error',
                           'x_cov_mat', 'y_data_cov_mat', 'y_model_cov_mat', 'total_cov_mat',
                           'x_cor_mat', 'y_data_cor_mat', 'y_model_cor_mat', 'total_cor_mat',
                           'x_cov_mat_inverse', 'y_data_cov_mat_inverse', 'y_model_cov_mat_inverse', 'total_cor_mat_inverse'
                           'x_data_cov_mat'}
    _BASIC_ERROR_NAMES = {
        'x_data_error', 'x_model_error', 'x_data_cov_mat', 'x_model_cov_mat',
        'y_data_error', 'y_model_error', 'y_data_cov_mat', 'y_model_cov_mat'
    }
    X_ERROR_ALGORITHMS = ('iterative linear', 'nonlinear')
    _STRING_TO_COST_FUNCTION = STRING_TO_COST_FUNCTION
    _AXES = (None, "x", "y")
    _MODEL_NAME = "y_model"
    _MODEL_ERROR_NODE_NAMES = ["y_model_error", "y_model_cov_mat"]
    _PROJECTED_NODE_NAMES = ["total_error", "total_cov_mat"]

    def __init__(self,
                 xy_data,
                 model_function=function_library.linear_model,
                 cost_function=XYCostFunction_Chi2(
                    axes_to_use='xy', errors_to_use='covariance'),
                 minimizer=None, minimizer_kwargs=None,
                 dynamic_error_algorithm="nonlinear"):
        """
        Construct a fit of a model to *xy* data.

        :param xy_data: the x and y measurement values
        :type xy_data: (2, N)-array of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe2.fit.xy.XYModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        :param minimizer: the minimizer to use for fitting.
        :type minimizer: None, "iminuit", "tminuit", or "scipy".
        :param minimizer_kwargs: dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """
        super(XYFit, self).__init__(
            data=xy_data, model_function=model_function, cost_function=cost_function,
            minimizer=minimizer, minimizer_kwargs=minimizer_kwargs,
            dynamic_error_algorithm=dynamic_error_algorithm)

    # -- private methods

    def _init_nexus(self):
        super(XYFit, self)._init_nexus()

        self._nexus.add_dependency(
            'total_cov_mat',
            depends_on=(
                'parameter_values',
                'x_model',
                'x_total_cov_mat',
                'y_total_cov_mat'
            )
        )
        self._nexus.add_dependency(
            'total_error',
            depends_on=(
                'parameter_values',
                'x_model',
                'x_total_error',
                'y_total_error'
            )
        )

        self._nexus.add_dependency(
            'y_model',
            depends_on=(
                'x_model',
                'parameter_values'
            )
        )

        self._nexus.add_dependency(
            'x_model',
            depends_on=(
                'x_data',
            )
        )

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise XYFitException("Incompatible container type '%s' (expected '%s')"
                                 % (type(new_data), self.CONTAINER_TYPE))
        else:
            _x_data = new_data[0]
            _y_data = new_data[1]
            self._data_container = XYContainer(_x_data, _y_data, dtype=float)
        self._data_container._on_error_change_callback = self._on_error_change

        # update nexus data nodes
        self._nexus.get('x_data').mark_for_update()
        self._nexus.get('y_data').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = XYParametricModel(
            self.x_model,
            self._model_function,
            self.parameter_values
        )

    def _report_data(self, output_stream, indent, indentation_level):
        output_stream.write(indent * indentation_level + '########\n')
        output_stream.write(indent * indentation_level + '# Data #\n')
        output_stream.write(indent * indentation_level + '########\n\n')
        _data_table_dict = OrderedDict()
        _data_table_dict['X Data'] = self.x_data
        if self._data_container.has_x_errors:
            _data_table_dict['X Data Error'] = self.x_data_error
            _data_table_dict['X Data Correlation Matrix'] = self.x_data_cor_mat

        print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

        _data_table_dict = OrderedDict()
        _data_table_dict['Y Data'] = self.y_data
        if self._data_container.has_y_errors:
            _data_table_dict['Y Data Error'] = self.y_data_error
            _data_table_dict['Y Data Correlation Matrix'] = self.y_data_cor_mat

        print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

    def _report_model(self, output_stream, indent, indentation_level):
        # call base method to show header and model function
        super(XYFit, self)._report_model(output_stream, indent, indentation_level)
        _model_table_dict = OrderedDict()
        _model_table_dict['X Model'] = self.x_model
        if self._param_model.has_x_errors:
            _model_table_dict['X Model Error'] = self.x_model_error
            _model_table_dict['X Model Correlation Matrix'] = self.x_model_cor_mat

        print_dict_as_table(_model_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

        _model_table_dict = OrderedDict()
        _model_table_dict['Y Model'] = self.y_model
        if self._param_model.has_y_errors:
            _model_table_dict['Y Model Error'] = self.y_model_error
            _model_table_dict['Y Model Correlation Matrix'] = self.y_model_cor_mat

        print_dict_as_table(_model_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')
        if self._param_model.get_matching_errors({"relative": True, "axis": 1}):
            output_stream.write(indent * (indentation_level + 1))
            output_stream.write(
                "y model covariance matrix was calculated dynamically relative to y model values.\n"
            )
            output_stream.write("\n")

    def _project_x_onto_y(self, x, y, sqrt=False):
        if x.ndim not in (1, 2):
            raise ValueError
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        if np.all(x == 0):
            return y

        _diag = x if x.ndim == 1 else np.diag(x)
        _precision = 0.01 * np.min(_diag) if sqrt else 0.01 * np.min(np.sqrt(_diag))
        _derivatives = self._param_model.eval_model_function_derivative_by_x(
            dx=_precision,
            model_parameters=self.parameter_values
        )
        _x_scale = _derivatives ** 2 if x.ndim == 1 else np.outer(_derivatives, _derivatives)
        if sqrt:
            return np.sqrt(y ** 2 + x ** 2 * _x_scale)
        else:
            return y + x * _x_scale

    def _set_data_as_model_ref(self):
        _errs_and_old_refs = []
        for _err in self._param_model.get_matching_errors({"relative": True, "axis": 1}).values():
            _old_ref = _err.reference
            _err.reference = self._data_container.y
            _errs_and_old_refs.append((_err, _old_ref))
        return _errs_and_old_refs

    def _iterative_fits_needed(self):
        return (bool(self._param_model.get_matching_errors({"relative": True, "axis": 1}))
                or self.has_x_errors) \
               and self._dynamic_error_algorithm == "iterative"

    def _second_fit_needed(self):
        return bool(self._param_model.get_matching_errors({"relative": True, "axis": 1})) \
               and self._dynamic_error_algorithm == "iterative"

    def _get_node_names_to_freeze(self, first_fit):
        if not self.has_x_errors or self._dynamic_error_algorithm == "iterative":
            return self._PROJECTED_NODE_NAMES + super(
                XYFit, self)._get_node_names_to_freeze(first_fit)
        else:
            return super(XYFit, self)._get_node_names_to_freeze(first_fit)

    # -- public properties

    @property
    def has_x_errors(self):
        """``True`` if at least one *x* uncertainty source has been defined"""
        return self._data_container.has_x_errors or self._param_model.has_x_errors

    @property
    def has_y_errors(self):
        """``True`` if at least one *y* uncertainty source has been defined"""
        return self._data_container.has_y_errors or self._param_model.has_y_errors

    @property
    def x_data(self):
        """array of measurement *x* values"""
        return self._data_container.x

    @property
    def x_model(self):
        return self.x_data

    @property
    def y_data(self):
        """array of measurement data *y* values"""
        return self._data_container.y

    @property
    def model(self):
        """(2, N)-array containing *x* and *y* model values"""
        return self._param_model.data

    @property
    def x_data_error(self):
        """array of pointwise *x* data uncertainties"""
        return self._data_container.x_err

    @property
    def y_data_error(self):
        """array of pointwise *y* data uncertainties"""
        return self._data_container.y_err

    @property
    def data_error(self):
        """array of pointwise *xy* uncertainties (projected onto the *y* axis)"""
        return self._project_x_onto_y(x=self.x_data_error, y=self.y_data_error, sqrt=True)

    @property
    def x_data_cov_mat(self):
        """the data *x* covariance matrix"""
        return self._data_container.x_cov_mat

    @property
    def y_data_cov_mat(self):
        """the data *y* covariance matrix"""
        return self._data_container.y_cov_mat

    @property
    def data_cov_mat(self):
        """the data *xy* covariance matrix (projected onto the *y* axis)"""
        return self._project_x_onto_y(x=self.x_data_cov_mat, y=self.y_data_cov_mat, sqrt=False)

    @property
    def x_data_cov_mat_inverse(self):
        """inverse of the data *x* covariance matrix (or ``None`` if singular)"""
        return self._data_container.x_cov_mat_inverse

    @property
    def y_data_cov_mat_inverse(self):
        """inverse of the data *y* covariance matrix (or ``None`` if singular)"""
        return self._data_container.y_cov_mat_inverse

    @property
    def data_cov_mat_inverse(self):
        """
        inverse of the data *xy* covariance matrix (projected onto the *y* axis, ``None`` if
        singular)
        """
        return invert_matrix(self.data_cov_mat)

    @property
    def x_data_cor_mat(self):
        """the data *x* correlation matrix"""
        return self._data_container.x_cor_mat

    @property
    def y_data_cor_mat(self):
        """the data *y* correlation matrix"""
        return self._data_container.y_cor_mat

    @property
    def data_cor_mat(self):
        """the data *xy* correlation matrix (projected onto the *y* axis)"""
        return CovMat(self.data_cov_mat).cor_mat

    @property
    def y_model(self):
        """array of *y* model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y

    @property
    def x_model_error(self):
        """array of pointwise model *x* uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_err

    @property
    def y_model_error(self):
        """array of pointwise model *y* uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_err

    @property
    def model_error(self):
        """array of pointwise model *xy* uncertainties (projected onto the *y* axis)"""
        return self._project_x_onto_y(x=self.x_model_error, y=self.y_model_error, sqrt=True)

    @property
    def x_model_cov_mat(self):
        """the model *x* covariance matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cov_mat

    @property
    def y_model_cov_mat(self):
        """the model *y* covariance matrix"""
        self._param_model.parameters = self.parameter_values # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cov_mat

    @property
    def model_cov_mat(self):
        """the model *xy* covariance matrix (projected onto the *y* axis)"""
        return self._project_x_onto_y(x=self.x_model_cov_mat, y=self.y_model_cov_mat, sqrt=False)

    @property
    def x_model_cov_mat_inverse(self):
        """inverse of the model *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cov_mat_inverse

    @property
    def y_model_cov_mat_inverse(self):
        """inverse of the model *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cov_mat_inverse

    @property
    def model_cov_mat_inverse(self):
        """
        inverse of the model *xy* covariance matrix (projected onto the *y* axis, ``None`` if
        singular)
        """
        return invert_matrix(self.model_cov_mat)

    @property
    def x_model_cor_mat(self):
        """the model *x* correlation matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cor_mat

    @property
    def y_model_cor_mat(self):
        """the model *y* correlation matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cor_mat

    @property
    def model_cor_mat(self):
        """the model *xy* correlation matrix (projected onto the *y* axis)"""
        return CovMat(self.model_cov_mat).cor_mat

    @property
    def x_total_error(self):
        """array of pointwise total *x* uncertainties"""
        return add_in_quadrature(self.x_model_error, self.x_data_error)

    @property
    def y_total_error(self):
        """array of pointwise total *y* uncertainties"""
        return add_in_quadrature(self.y_model_error, self.y_data_error)

    @property
    def total_error(self):
        """array of pointwise total *xy* uncertainties (projected onto the *y* axis)"""
        return self._project_x_onto_y(x=self.x_total_error, y=self.y_total_error, sqrt=True)

    @property
    def x_total_cov_mat(self):
        """the total *x* covariance matrix"""
        return self.x_data_cov_mat + self.x_model_cov_mat

    @property
    def y_total_cov_mat(self):
        """the total *y* covariance matrix"""
        return self.y_data_cov_mat + self.y_model_cov_mat

    @property
    def total_cov_mat(self):
        """the total *xy* covariance matrix (projected onto the *y* axis)"""
        return self._project_x_onto_y(x=self.x_total_cov_mat, y=self.y_total_cov_mat, sqrt=False)

    @property
    def x_total_cov_mat_inverse(self):
        """inverse of the total *x* covariance matrix (or ``None`` if singular)"""
        return invert_matrix(self.x_total_cov_mat)

    @property
    def y_total_cov_mat_inverse(self):
        """inverse of the total *y* covariance matrix (or ``None`` if singular)"""
        return invert_matrix(self.y_total_cov_mat)

    @property
    def total_cov_mat_inverse(self):
        """
        inverse of the total *xy* covariance matrix (projected onto the *y* axis, ``None`` if
        singular)
        """
        return invert_matrix(self.total_cov_mat)

    @property
    def x_total_cor_mat(self):
        """the total *x* correlation matrix"""
        return CovMat(self.x_total_cov_mat).cor_mat

    @property
    def y_total_cor_mat(self):
        """the total *y* correlation matrix"""
        return CovMat(self.y_total_cov_mat).cor_mat

    @property
    def x_range(self):
        """range of the *x* measurement data"""
        return self._data_container.x_range

    @property
    def y_range(self):
        """range of the *y* measurement data"""
        return self._data_container.y_range

    # -- public methods

    def add_error(self, axis, err_val,
                  name=None, correlation=0, relative=False, reference='data'):
        """
        Add an uncertainty source for axis to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :param reference: which reference values to use when calculating absolute errors from relative errors
        :type reference: 'data' or 'model'
        :return: error id
        :rtype: int
        """
        return super(XYFit, self).add_error(err_val=err_val,
                                            name=name,
                                            correlation=correlation,
                                            relative=relative,
                                            reference=reference,
                                            axis=axis)

    def add_matrix_error(self, axis, err_matrix, matrix_type,
                         name=None, err_val=None, relative=False, reference='data'):
        """
        Add a matrix uncertainty source for an axis to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :param reference: which reference values to use when calculating absolute errors from relative errors
        :type reference: 'data' or 'model'
        :return: error id
        :rtype: int
        """
        return super(XYFit, self).add_matrix_error(err_matrix=err_matrix,
                                                   matrix_type=matrix_type,
                                                   name=name,
                                                   err_val=err_val,
                                                   relative=relative,
                                                   reference=reference,
                                                   axis=axis)

    def eval_model_function(self, x=None, model_parameters=None):
        """
        Evaluate the model function.

        :param x: values of *x* at which to evaluate the model function (if ``None``, the data *x* values are used)
        :type x: iterable of float
        :param model_parameters: the model parameter values (if ``None``, the current values are used)
        :type model_parameters: iterable of float
        :return: model function values
        :rtype: :py:class:`numpy.ndarray`
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.eval_model_function(x=x, model_parameters=model_parameters)

    def eval_model_function_derivative_by_parameters(self, x=None, model_parameters=None):
        """
        Evaluate the model function derivative for each parameter.

        :param x: values of *x* at which to evaluate the model function (if ``None``, the data *x* values are used)
        :type x: iterable of float
        :param model_parameters: the model parameter values (if ``None``, the current values are used)
        :type model_parameters: iterable of float
        :return: model function values
        :rtype: :py:class:`numpy.ndarray`
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.eval_model_function_derivative_by_parameters(x=x, model_parameters=model_parameters)
