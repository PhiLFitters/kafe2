try:
    import typing  # help IDEs with type-hinting inside docstrings
except ImportError:
    pass
import numpy  # help IDEs with type-hinting inside docstrings
from collections import OrderedDict
from copy import deepcopy
import numpy as np

from ...core.error import CovMat
from ...tools import print_dict_as_table
from .._base import FitException, FitBase, DataContainerBase, ModelFunctionBase
from .container import XYContainer
from .cost import XYCostFunction_Chi2, STRING_TO_COST_FUNCTION
from .model import XYParametricModel
from .plot import XYPlotAdapter
from ..util import function_library, add_in_quadrature, invert_matrix


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
        """Construct a fit of a model to *xy* data.

        :param xy_data: A :py:obj:`~.XYContainer` or a raw 2D array of shape ``(2, N)``
         containing the measurement data.
        :type xy_data: XYContainer or typing.Sequence
        :param model_function: The model function as a native Python function where the first
            argument denotes the independent *x* variable or an already defined
            :py:class:`~kafe2.fit.xy.XYModelFunction` object.
        :type model_function: typing.Callable
        :param cost_function: The cost function this fit uses to find the best parameters.
        :type cost_function: str or typing.Callable
        :param minimizer: The minimizer to use for fitting. Either :py:obj:`None`, ``"iminuit"``,
            ``"tminuit"``, or ``"scipy"``.
        :type minimizer: str or None
        :param minimizer_kwargs: Dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """
        super(XYFit, self).__init__(
            data=xy_data, model_function=model_function, cost_function=cost_function,
            minimizer=minimizer, minimizer_kwargs=minimizer_kwargs,
            dynamic_error_algorithm=dynamic_error_algorithm)

    # -- private methods

    def _init_nexus(self):
        super(XYFit, self)._init_nexus()

        self._nexus.add_function(
            func=self._project_cov_mat,
            func_name="total_cov_mat",
            par_names=[
                "x_total_cov_mat",
                "y_total_cov_mat",
                "x_model",
                "parameter_values"
            ],
            existing_behavior="replace"
        )
        self._nexus.add_function(
            func=self._project_error,
            func_name="total_error",
            par_names=[
                "x_total_error",
                "y_total_error",
                "x_model",
                "parameter_values"
            ],
            existing_behavior="replace"
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

    def _project_cov_mat(self, x_cov_mat, y_cov_mat, x_model, parameter_values):
        _diag = np.diag(x_cov_mat)
        if np.all(_diag == 0):
            return y_cov_mat
        _derivatives = self._param_model.eval_model_function_derivative_by_x(
            x=x_model,
            dx=0.01 * np.sqrt(_diag),
            model_parameters=parameter_values
        )
        return y_cov_mat + x_cov_mat * np.outer(_derivatives, _derivatives)

    def _project_error(self, x_error, y_error, x_model, parameter_values):
        if np.all(x_error == 0):
            return y_error
        _derivatives = self._param_model.eval_model_function_derivative_by_x(
            x=x_model,
            dx=0.01 * x_error,
            model_parameters=parameter_values
        )
        return np.sqrt(np.square(y_error) + np.square(x_error * _derivatives))

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
               and self._dynamic_error_algorithm == "nonlinear"

    def _get_node_names_to_freeze(self, first_fit):
        if not self.has_x_errors or self._dynamic_error_algorithm == "iterative":
            return self._PROJECTED_NODE_NAMES + super(
                XYFit, self)._get_node_names_to_freeze(first_fit)
        else:
            return super(XYFit, self)._get_node_names_to_freeze(first_fit)

    # -- public properties

    @property
    def has_x_errors(self):
        """:py:obj:`True`` if at least one *x* uncertainty source has been defined.

        :rtype: bool
        """
        return self._data_container.has_x_errors or self._param_model.has_x_errors

    @property
    def has_y_errors(self):
        """:py:obj:`True`` if at least one *y* uncertainty source has been defined

        :rtype: bool
        """
        return self._data_container.has_y_errors or self._param_model.has_y_errors

    @property
    def x_data(self):
        """1D array containing the measurement *x* values.

        :rtype: numpy.ndarray[float]
        """
        return self._data_container.x

    @property
    def x_model(self):
        """1D array containing the model *x* values. The same as :py;obj:`.x_data` for an
            :py:obj:`~.XYFit`.

        :rtype: numpy.ndarray[float]
        """
        return self.x_data

    @property
    def y_data(self):
        """1D array containing the measurement *y* values.

        :rtype: numpy.ndarray[float]
        """
        return self._data_container.y

    @property
    def model(self):
        """2D array of shape ``(2, N)`` containing the *x* and *y* model values

        :rtype: numpy.ndarray
        """
        return self._param_model.data

    @property
    def x_data_error(self):
        """1D array containing the pointwise *x* data uncertainties

        :rtype: numpy.ndarray[float]
        """
        return self._data_container.x_err

    @property
    def y_data_error(self):
        """1D array containing the pointwise *y* data uncertainties

        :rtype: numpy.ndarray[float]
        """
        return self._data_container.y_err

    @property
    def data_error(self):
        """1D array containing the pointwise *xy* uncertainties projected onto the *y* axis.

        :rtype: numpy.ndarray[float]
        """
        return self._project_error(
            self.x_data_error, self.y_data_error, self.x_model, self.parameter_values)

    @property
    def x_data_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the data *x* covariance matrix.

        :rtype: numpy.ndarray
        """
        return self._data_container.x_cov_mat

    @property
    def y_data_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the data *y* covariance matrix.

        :rtype: numpy.ndarray
        """
        return self._data_container.y_cov_mat

    @property
    def data_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the data *xy* covariance matrix (projected
        onto the *y* axis).

        :rtype: numpy.ndarray
        """
        return self._project_cov_mat(
            self.x_data_cov_mat, self.y_data_cov_mat, self.x_model, self.parameter_values)

    @property
    def x_data_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing the inverse of the data *x* covariance matrix or
        :py:obj:`None` if singular.

        :rtype: numpy.ndarray or None
        """
        return self._data_container.x_cov_mat_inverse

    @property
    def y_data_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing the inverse of the data *y* covariance matrix or
        :py:obj:`None` if singular.

        :rtype: numpy.ndarray or None
        """
        return self._data_container.y_cov_mat_inverse

    @property
    def data_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing the inverse of the data *xy* covariance matrix
        projected onto the *y* axis. :py:obj:`None` if singular.

        :rtype: numpy.ndarray or None
        """
        return invert_matrix(self.data_cov_mat)

    @property
    def x_data_cor_mat(self):
        """2D array of shape ``(N, N)`` containing the data *x* correlation matrix.

        :rtype: numpy.ndarray
        """
        return self._data_container.x_cor_mat

    @property
    def y_data_cor_mat(self):
        """2D array of shape ``(N, N)`` containing the data *y* correlation matrix.

        :rtype: numpy.ndarray
        """
        return self._data_container.y_cor_mat

    @property
    def data_cor_mat(self):
        """2D array of shape ``(N, N)`` containing the data *xy* correlation matrix projected
        onto the *y* axis.

        :rtype: numpy.ndarray
        """
        return CovMat(self.data_cov_mat).cor_mat

    @property
    def y_model(self):
        """1D array of *y* model predictions for the data points.

        :rtype: numpy.ndarray[float]
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y

    @property
    def x_model_error(self):
        """1D array of pointwise model *x* uncertainties.

        :rtype: numpy.ndarray[float]
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_err

    @property
    def y_model_error(self):
        """1D array of pointwise model *y* uncertainties.

        :rtype: numpy.ndarray[float]
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_err

    @property
    def model_error(self):
        """1D array of pointwise model *xy* uncertainties projected onto the *y* axis.

        :rtype: numpy.ndarray[float]
        """
        return self._project_error(
            self.x_model_error, self.y_model_error, self.x_model, self.parameter_values)

    @property
    def x_model_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the model *x* covariance matrix.

        :rtype: numpy.ndarray
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cov_mat

    @property
    def y_model_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the model *y* covariance matrix.

        :rtype: numpy.ndarray
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cov_mat

    @property
    def model_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the model *xy* covariance matrix projected onto
        the *y* axis.

        :rtype: numpy.ndarray
        """
        return self._project_cov_mat(
            self.x_model_cov_mat, self.y_model_cov_mat, self.x_model, self.parameter_values)

    @property
    def x_model_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing the inverse of the model *x* covariance matrix
        or :py:obj:`None` if singular.

        :rtype: numpy.ndarray or None
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cov_mat_inverse

    @property
    def y_model_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing the inverse of the model *y* covariance matrix
        or :py:obj:`None` if singular.

        :rtype: numpy.ndarray
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cov_mat_inverse

    @property
    def model_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing the inverse of the model *xy* covariance matrix
        projected onto the *y* axis. :py:obj:`None`` if singular.

        :rtype: numpy.ndarray
        """
        return invert_matrix(self.model_cov_mat)

    @property
    def x_model_cor_mat(self):
        """2D array of shape ``(N, N)`` containing the model *x* correlation matrix.

        :rtype: numpy.ndarray
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cor_mat

    @property
    def y_model_cor_mat(self):
        """2D array of shape ``(N, N)`` containing the model *y* correlation matrix.

        :rtype: numpy.ndarray
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cor_mat

    @property
    def model_cor_mat(self):
        """2D array of shape ``(N, N)`` containing the model *xy* correlation matrix projected onto
        the *y* axis.

        :rtype: numpy.ndarray
        """
        return CovMat(self.model_cov_mat).cor_mat

    @property
    def x_total_error(self):
        """1D array of total pointwise *x* uncertainties.

        :rtype: numpy.ndarray[float]
        """
        return self._nexus.get("x_total_error").value

    @property
    def y_total_error(self):
        """1D array of total pointwise *y* uncertainties

        :rtype: numpy.ndarray[float]
        """
        return self._nexus.get("y_total_error").value

    @property
    def total_error(self):
        """1D array of the total pointwise *xy* uncertainties projected onto the *y* axis.

        :rtype: numpy.ndarray[float]
        """
        return self._nexus.get("total_error").value

    @property
    def x_total_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the total *x* covariance matrix.

        :rtype: numpy.ndarray
        """
        return self._nexus.get("x_total_cov_mat").value

    @property
    def y_total_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the total *y* covariance matrix.

        :rtype: numpy.ndarray
        """
        return self._nexus.get("y_total_cov_mat").value

    @property
    def total_cov_mat(self):
        """2D array of shape ``(N, N)`` containing the total *xy* covariance matrix projected onto
        the *y* axis.

        :rtype: numpy.ndarray
        """
        return self._nexus.get("total_cov_mat").value

    @property
    def x_total_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing inverse of the total *x* covariance matrix.
        :py:obj:`None` if singular.

        :rtype: numpy.ndarray
        """
        return invert_matrix(self.x_total_cov_mat)

    @property
    def y_total_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing inverse of the total *y* covariance matrix.
        :py:obj:`None` if singular.

        :rtype: numpy.ndarray
        """
        return invert_matrix(self.y_total_cov_mat)

    @property
    def total_cov_mat_inverse(self):
        """2D array of shape ``(N, N)`` containing theinverse of the total *xy* covariance matrix
        projected onto the *y* axis. :py:obj:`None` if singular.

        :rtype: numpy.ndarray
        """
        return invert_matrix(self.total_cov_mat)

    @property
    def x_total_cor_mat(self):
        """2D array of shape ``(N, N)`` containing the total *x* correlation matrix.

        :rtype: numpy.ndarray
        """
        return CovMat(self.x_total_cov_mat).cor_mat

    @property
    def y_total_cor_mat(self):
        """2D array of shape ``(N, N)`` containing the total *y* correlation matrix.

        :rtype: numpy.ndarray
        """
        return CovMat(self.y_total_cov_mat).cor_mat

    @property
    def x_range(self):
        """Minimum and maximum values of the *x* measurement data.

        :rtype: tuple[float, float]
        """
        return self._data_container.x_range

    @property
    def y_range(self):
        """Minimum and maximum values of the *y* measurement data.

        :rtype: tuple[float, float]
        """
        return self._data_container.y_range

    # -- public methods

    def add_error(self, axis, err_val,
                  name=None, correlation=0, relative=False, reference='data'):
        """Add an uncertainty source for an axis to the data container.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_val: Pointwise uncertainties or a single uncertainty for all data points.
        :type err_val: float or typing.Sequence[float]
        :param name: Unique name for this uncertainty source. If :py:obj:`None`, the name
            of the error source will be set to a random alphanumeric string.
        :type name: str or None
        :param correlation: Correlation coefficient between any two distinct data points.
        :type correlation: float
        :param relative: If :py:obj:`True`, **err_val** will be interpreted as a *relative*
            uncertainty.
        :type relative: bool
        :param reference: Which reference values to use when calculating absolute errors from
            relative errors. Either ``'data'`` or ``'model'``.
        :type reference: str
        :return: An error id uniquely identifying the created error source.
        :rtype: str
        """
        return super(XYFit, self).add_error(err_val=err_val,
                                            name=name,
                                            correlation=correlation,
                                            relative=relative,
                                            reference=reference,
                                            axis=axis)

    def add_matrix_error(self, axis, err_matrix, matrix_type,
                         name=None, err_val=None, relative=False, reference='data'):
        """Add a matrix uncertainty source for an axis to the data container.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_matrix: 2D array of shape ``(size, size)`` containing the covariance or
            correlation matrix
        :type err_matrix: numpy.ndarray
        :param matrix_type: One of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``.
        :type matrix_type: str
        :param name: Unique name for this uncertainty source. If :py:obj:`None`, the name
            of the error source will be set to a random alphanumeric string.
        :type name: str or None
        :param err_val: The pointwise uncertainties. This is mandatory if only a correlation
            matrix is given.
        :type err_val: typing.Sequence[float]
        :param relative: If :py:obj:`True`, the covariance matrix and/or **err_val** will be
            interpreted as a *relative* uncertainty.
        :type relative: bool
        :param reference: Which reference values to use when calculating absolute errors from
            relative errors. Either ``'data'`` or ``'model'``.
        :type reference: str
        :return: An error id uniquely identifying the created error source.
        :rtype: str
        """
        return super(XYFit, self).add_matrix_error(err_matrix=err_matrix,
                                                   matrix_type=matrix_type,
                                                   name=name,
                                                   err_val=err_val,
                                                   relative=relative,
                                                   reference=reference,
                                                   axis=axis)

    def eval_model_function(self, x=None, model_parameters=None):
        """Evaluate the model function.

        :param x: 1D array containing the values of *x* at which to evaluate the model function. If
            :py:obj:`None`, the data *x* values :py:obj:`~XYFit.x_data` are used.
        :type x: numpy.ndarray[float]
        :param model_parameters: The model parameter values. If :py:obj:`None`, the current values
            :py:obj:`~.XYFit.parameter_values` are used.
        :type model_parameters: typing.Collection[float]
        :return: Model function values at the given *x*-values.
        :rtype: numpy.ndarray[float]
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.eval_model_function(x=x, model_parameters=model_parameters)

    def eval_model_function_derivative_by_parameters(self, x=None, model_parameters=None, par_dx=None):
        """Evaluate the derivative of the model function with respect to the model parameters.

        :param x: 1D array containing the *x* values at which to evaluate the model function. If
            :py:obj:`None`, the data *x* values :py:obj:`~XYFit.x_data` are used.
        :type x: numpy.ndarray[float]
        :param model_parameters: 1D array containing the model parameter values. If :py:obj:`None`,
            the current values :py:obj:`~XYFit.parameter_values` are used.
        :type model_parameters: typing.Collection[float]
        :param par_dx: 1D array with length ``pars`` containing the numeric differentiation step
            size for each parameter. If :py:obj:`None` and a fit has been performed, 1% of the parameter uncertainties
            is used.
        :type par_dx: typing.Collection[float]
        :return: 2D array of shape ``(par, N)`` containing the model function derivatives for
            each parameter at the given *x* values.
        :rtype: numpy.ndarray[numpy.ndarray[float]]
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if par_dx is None and self.did_fit:
            par_dx = 1e-2*self.parameter_errors
        return self._param_model.eval_model_function_derivative_by_parameters(
            x=x, model_parameters=model_parameters, par_dx=par_dx)

    def error_band(self, x=None):
        """Calculate the symmetric model uncertainty at every given point *x*. This is only
        possible after a fit has been performed with the :py:meth:`.do_fit` method.

        :param numpy.ndarray[float] x: 1D array containing the values of *x* at which to calculate
            the model uncertainty.
        :return: 1D array containing the model uncertainties at the given *x* values.
        :rtype: numpy.ndarray[float]
        """
        if not self.did_fit:
            raise XYFitException('Cannot calculate an error band without first performing a fit.')
        if x is None:
            x = self.x_model
        if self.parameter_cov_mat is None:
            return np.zeros_like(x)

        _f_deriv_by_params = self.eval_model_function_derivative_by_parameters(x=x)
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[par_idx][x_idx]

        _f_deriv_by_params = _f_deriv_by_params.T
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[x_idx][par_idx]

        _band_y = np.zeros_like(x)

        # Cut out fixed parameters which have nan as derivative:
        _not_pars_fixed = [_par_name not in self._fitter.fixed_parameters
                           for _par_name in self.parameter_names]
        _cut_parameter_cov_mat = self.parameter_cov_mat[_not_pars_fixed][:, _not_pars_fixed]
        for _x_idx, _x_val in enumerate(x):
            _p_res = _f_deriv_by_params[_x_idx, _not_pars_fixed]
            _band_y[_x_idx] = _p_res.dot(_cut_parameter_cov_mat).dot(_p_res)

        return np.sqrt(_band_y)
