from collections import OrderedDict
from copy import deepcopy

import numpy as np
import six
import sys
import textwrap

from ...tools import print_dict_as_table
from ...core import NexusFitter, Nexus
from ...core.fitters.nexus import Parameter, Alias
from ...config import kc
from .._base import FitException, FitBase, DataContainerBase, CostFunctionBase
from ..xy.fit import XYFit
from .container import XYMultiContainer
from .cost import XYMultiCostFunction_Chi2, XYMultiCostFunction_UserDefined, STRING_TO_COST_FUNCTION
from .model import XYMultiParametricModel, XYMultiModelFunction
#from .plot import XYMultiPlotAdapter  # TODO: reimplement
from ..util import function_library, add_in_quadrature, collect, invert_matrix


__all__ = ["XYMultiFit"]


class XYMultiFitException(FitException):
    pass


class XYMultiFit(FitBase):
    CONTAINER_TYPE = XYMultiContainer
    MODEL_TYPE = XYMultiParametricModel
    MODEL_FUNCTION_TYPE = XYMultiModelFunction
    PLOT_ADAPTER_TYPE = None  # TODO: re-configure this when implemented
    EXCEPTION_TYPE = XYMultiFitException
    RESERVED_NODE_NAMES = {'y_data', 'y_model', 'cost',
                           'x_error', 'y_data_error', 'y_model_error', 'total_error',
                           'x_cov_mat', 'y_data_cov_mat', 'y_model_cov_mat', 'total_cov_mat',
                           'x_cor_mat', 'y_data_cor_mat', 'y_model_cor_mat', 'total_cor_mat',
                           'x_cov_mat_inverse', 'y_data_cov_mat_inverse', 'y_model_cov_mat_inverse', 'total_cor_mat_inverse'
                           'y_data_uncor_cov_mat', 'y_model_uncor_cov_mat','y_total_uncor_cov_mat',
                           'nuisance_y_data_cor_cov_mat','nuisance_y_model_cor_cov_mat','nuisance_y_total_cor_cov_mat',
                           'nuisance_para', 'y_nuisance_vector'}

    def __init__(self,
                 xy_data,
                 model_function,
                 cost_function=XYMultiCostFunction_Chi2(
                    axes_to_use='xy',
                    errors_to_use='covariance'),
                 x_error_algorithm='nonlinear',
                 minimizer=None,
                 minimizer_kwargs=None):
        """
        Construct a fit of one or more pairs of models and *xy* datasets.
        To specify more than one such pair, xy_data and model_function must be iterables of the same length.
        The model function at each index is associated with the dataset at the same index.

        :param xy_data: the x and y measurement values
        :type xy_data: (2, N)-array of float or an iterable thereof
        :param model_function: the model function
        :type model_function: :py:class:`~kafe2.fit.multi.XYMultiModelFunction` or unwrapped native Python function or an
            iterable of those
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        :param minimizer: the minimizer to be used for the fit, 'root', or 'tminuit' for TMinuit, 'iminuit' for IMinuit, or 'scipy' for SciPy.
            If None, the minimizer will be chosen according to config (TMinuit > IMinuit > SciPy by default)
        :type str
        :param minimizer_kwargs: kwargs provided to the minimizer constructor
        :type minimizer_kwargs: native Python dictionary
        """
        # set the cost function, validation is done when setting the data
        if isinstance(cost_function, CostFunctionBase):
            self._cost_function = cost_function
        elif isinstance(cost_function, str):
            _cost_function_class = STRING_TO_COST_FUNCTION.get(cost_function, None)
            if _cost_function_class is None:
                raise XYMultiFitException('Unknown cost function: %s' % cost_function)
            self._cost_function = _cost_function_class()
        else:
            self._cost_function = XYMultiCostFunction_UserDefined(cost_function)
            # self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?

        # validate x error algorithm
        if x_error_algorithm not in XYFit.X_ERROR_ALGORITHMS:
            raise ValueError(
                "Unknown value for 'x_error_algorithm': "
                "{}. Expected one of:".format(
                    x_error_algorithm,
                    ', '.join(['iterative linear', 'nonlinear'])
                )
            )
        else:
            self._x_error_algorithm = x_error_algorithm

        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs

        # constructing the model function needs the data container
        self._set_new_data(xy_data)

        # set/construct the model function object
        if isinstance(model_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_function
            self._model_function.data_indices = self._data_container.data_indices
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_function,
                                        self._data_container.data_indices)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        # initialize the Nexus
        self._init_nexus_callbacks = []
        self._init_nexus()

        # save minimizer, minimizer_kwargs for serialization
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs

        # initialize the Fitter
        self._initialize_fitter()
        # create the child ParametricModel object
        self._set_new_parametric_model()
        # TODO: check where to update this (set/release/etc.)
        # FIXME: nicer way than len()?
        self._cost_function.ndf = self._data_container.size - len(self._param_model.parameters)

        self._fit_param_constraints = []
        self._loaded_result_dict = None

    # -- private methods

    def _init_nexus(self):
        six.get_unbound_function(XYFit._init_nexus)(self)  # same nexus wiring as for simple XYFit
        for _callback in self._init_nexus_callbacks:
            _callback()

    def _calculate_y_error_band(self, num_points=100):
        # TODO: config for num_points
        _band_x = np.empty(self.model_count * num_points, dtype=float)
        for _i in range(self.model_count):
            _xmin, _xmax = self.get_x_range(_i)
            _band_x[_i * num_points : (_i + 1) * num_points] = np.linspace(_xmin, _xmax, num_points)
        _f_deriv_by_params = self._param_model.eval_model_function_derivative_by_parameters(x=_band_x,
                                x_indices=range(0, self.model_count * num_points + 1, num_points),
                                model_parameters=self.poi_values)
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[par_idx][x_idx]

        _f_deriv_by_params = _f_deriv_by_params.T
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[x_idx][par_idx]

        _band_y = np.zeros_like(_band_x)
        _n_poi = len(self.poi_values)
        for _x_idx, _x_val in enumerate(_band_x):
            _p_res = _f_deriv_by_params[_x_idx]
            _band_y[_x_idx] = _p_res.dot(self.parameter_cov_mat[:_n_poi, :_n_poi]).dot(_p_res)

        return np.sqrt(_band_y)

    def _get_poi_index_by_name(self, name):
        try:
            return self._poi_names.index(name)
        except ValueError:
            raise self.EXCEPTION_TYPE('Unknown parameter name: %s' % name)

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise XYMultiFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = self._new_data_container(new_data, dtype=float)
        # TODO: Think of a better way when setting new data to not always delete all labels
        self._axis_labels = [[None, None] for _ in range(self._data_container.num_datasets)]
        # mark nexus nodes for update
        # hasattr is needed, because data needs to be set before the nexus can be initialized
        if hasattr(self, '_nexus'):
            self._nexus.get('x_data').mark_for_update()
            self._nexus.get('y_data').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = self._new_parametric_model(
            self.x_model,
            self._model_function,
            self.poi_values
        )

    def _report_data(self, output_stream, indent, indentation_level):
        output_stream.write(indent * indentation_level + '########\n')
        output_stream.write(indent * indentation_level + '# Data #\n')
        output_stream.write(indent * indentation_level + '########\n\n')
        _data_table_dict = OrderedDict()
        _data_table_dict['X Data'] = self.x_data
        if self._data_container.has_x_errors:
            _data_table_dict['X Data Error'] = self.x_data_error
            _data_table_dict['X Data Total Correlation Matrix'] = self.x_data_cor_mat

        print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

        _data_table_dict = OrderedDict()
        _data_table_dict['Y Data'] = self.y_data
        if self.has_data_errors:
            _data_table_dict['Y Data Error'] = self.y_data_error
            _data_table_dict['Y Data Total Correlation Matrix'] = self.y_data_cor_mat

        print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

    def _report_model(self, output_stream, indent, indentation_level):
        output_stream.write(indent * indentation_level + '#########\n')
        output_stream.write(indent * indentation_level + '# Model #\n')
        output_stream.write(indent * indentation_level + '#########\n\n')

        for _i in range(self.model_count):
            output_stream.write(indent * (indentation_level + 1) + "Model Function\n")
            output_stream.write(indent * (indentation_level + 1) + "==============\n\n")
            output_stream.write(indent * (indentation_level + 2))
            output_stream.write(
                self._model_function.formatter.get_formatted(
                    model_index=_i,
                    with_par_values=False,
                    n_significant_digits=2,
                    format_as_latex=False,
                    with_expression=True
                )
            )
            output_stream.write('\n\n')
        output_stream.write('\n')

        _data_table_dict = OrderedDict()
        _data_table_dict['X Model'] = self.x_model
        if self.has_model_errors:
            _data_table_dict['X Model Error'] = self.x_model_error
            _data_table_dict['X Model Total Correlation Matrix'] = self.x_model_cor_mat

        print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

        _data_table_dict = OrderedDict()
        _data_table_dict['Y Model'] = self.y_model
        if self.has_model_errors:
            _data_table_dict['Y Model Error'] = self.y_model_error
            _data_table_dict['Y Model Total Correlation Matrix'] = self.y_model_cor_mat

        print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

    # -- public properties

    @property
    def x_data(self):
        """array of measurement *x* values"""
        return self._data_container.x

    @property
    def x_model(self):
        # if cost function uses x-nuisance parameters, consider these
        if self._cost_function.get_flag("need_x_nuisance") and self._data_container.has_uncor_x_errors:
            return self.x_data + (self.x_uncor_nuisance_values * self.x_data_error)
        else:
            return self.x_data

    @property
    def x_error(self):
        """array of pointwise *x* uncertainties"""
        return self._data_container.x_err

    @property
    def x_cov_mat(self):
        """the *x* covariance matrix"""
        return self._data_container.x_cov_mat

    @property
    def y_data(self):
        """array of measurement data *y* values"""
        return self._data_container.y

    @FitBase.data.getter
    def data(self):
        """(2, N)-array containing *x* and *y* measurement values"""
        return self._data_container.data

    @property
    def axis_labels(self):
        """the axis-labels to be passed on to the plot"""
        return self._axis_labels

    @axis_labels.setter
    def axis_labels(self, labels):
        """sets the axis-labels to be passed on to the plot

        :param labels: list of axis labels
        :type labels: list
        """
        if len(labels) != len(self._axis_labels) or len(labels[0]) != len(self._axis_labels[0]):
            raise XYMultiFitException("The dimensions of labels must fit the dimension of the data")
        self._axis_labels = labels

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
    def x_data_cov_mat(self):
        """the data *x* covariance matrix"""
        return self._data_container.x_cov_mat

    @property
    def y_data_cov_mat(self):
        """the data *y* covariance matrix"""
        return self._data_container.y_cov_mat

    @property
    def x_data_cov_mat_inverse(self):
        """inverse of the data *x* covariance matrix (or ``None`` if singular)"""
        return self._data_container.x_cov_mat_inverse

    @property
    def y_data_cov_mat_inverse(self):
        """inverse of the data *y* covariance matrix (or ``None`` if singular)"""
        return self._data_container.y_cov_mat_inverse

    @property
    def x_data_cor_mat(self):
        """the data *x* correlation matrix"""
        return self._data_container.x_cor_mat

    @property
    def y_data_uncor_cov_mat(self):
        """uncorrelated part of the data *y* covariance matrix (or ``None`` if singular)"""
        return self._data_container.y_uncor_cov_mat

    @property
    def y_data_uncor_cov_mat_inverse(self):
        """inverse of the uncorrelated part of the data *y* covariance matrix (or ``None`` if singular)"""
        return self._data_container.y_uncor_cov_mat_inverse

    @property
    def _y_data_nuisance_cor_design_mat(self):
        """matrix containing the correlated parts of all data uncertainties for all data points"""
        return self._data_container._y_nuisance_cor_design_mat

    @property
    def x_data_uncor_cov_mat(self):
        # data x uncorrelated covariance matrix
        return self._data_container.x_uncor_cov_mat

    @property
    def x_data_uncor_cov_mat_inverse(self):
        # data x uncorrelated inverse covariance matrix
        return self._data_container.x_uncor_cov_mat_inverse

    # TODO: correlated x-errors
    # @property
    # def _x_data_nuisance_cor_design_mat(self):
    #     # date x correlated matrix (nuisance)
    #     return self._data_container.nuisance_y_cor_cov_mat

    @property
    def y_data_cor_mat(self):
        """the data *y* correlation matrix"""
        return self._data_container.y_cor_mat

    @property
    def y_model(self):
        """array of *y* model predictions for the data points"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y

    @property
    def x_model_error(self):
        """array of pointwise model *x* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_err

    @property
    def y_model_error(self):
        """array of pointwise model *y* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_err

    @property
    def x_model_cov_mat(self):
        """the model *x* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cov_mat

    @property
    def y_model_cov_mat(self):
        """the model *y* covariance matrix"""
        self._param_model.parameters = self.poi_values # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cov_mat

    @property
    def x_model_cov_mat_inverse(self):
        """inverse of the model *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_cov_mat_inverse

    @property
    def y_model_cov_mat_inverse(self):
        """inverse of the model *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cov_mat_inverse

    @property
    def y_model_uncor_cov_mat(self):
        """uncorrelated part the model *y* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_uncor_cov_mat

    @property
    def y_model_uncor_cov_mat_inverse(self):
        """inverse of the uncorrelated part the model *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_uncor_cov_mat_inverse

    @property
    def x_model_uncor_cov_mat(self):
        """the model *x* uncorrelated covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_uncor_cov_mat

    @property
    def x_model_uncor_cov_mat_inverse(self):
        """inverse of the model *x*  uncorrelated covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.x_uncor_cov_mat_inverse

    @property
    def _y_model_nuisance_cor_design_mat(self):
        """matrix containing the correlated parts of all model uncertainties for all data points"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model._y_nuisance_cor_design_mat

    @property
    def x_model_cor_mat(self):
        """the model *x* correlation matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cor_mat

    # @property TODO: correlated x-errors
    # def _x_model_nuisance_cor_design_mat(self):
    #     """model *x*  correlated covariance matrix (nuisance) (or ``None`` if singular)"""
    #     self._param_model.parameters = self.poi_values  # this is lazy, so just do it
    #     self._param_model.x = self.x_with_errors
    #     return self._param_model.nuisance_x_cor_cov_mat

    @property
    def y_model_cor_mat(self):
        """the model *y* correlation matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.y_cor_mat

    @property
    def x_total_error(self):
        """array of pointwise total *x* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return add_in_quadrature(self.x_data_error, self.x_model_error)

    @property
    def y_total_error(self):
        """array of pointwise total *y* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return add_in_quadrature(self.y_data_error, self.y_model_error)

    @property
    def projected_xy_total_error(self):
        """array of pointwise total *y* with the x uncertainties projected on top of them"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_error

        _x_errors = self.x_total_error
        _precision = 0.01 * np.min(_x_errors)
        _derivatives = self._param_model.eval_model_function_derivative_by_x(dx=_precision)

        return np.sqrt(self.y_total_error**2 + self.x_total_error**2 * _derivatives**2)

    @property
    def x_total_cov_mat(self):
        """the total *x* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self.x_data_cov_mat + self.x_model_cov_mat

    @property
    def y_total_cov_mat(self):
        """the total *y* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self.y_data_cov_mat + self.y_model_cov_mat

    @property
    def projected_xy_total_cov_mat(self):
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_cov_mat

        _x_errors = self.x_total_error
        _precision = 0.01 * np.min(_x_errors)
        _derivatives = self._param_model.eval_model_function_derivative_by_x(dx=_precision)
        _outer_product = np.outer(_derivatives, _derivatives)

        return  self.y_total_cov_mat + self.x_total_cov_mat * _outer_product

    @property
    def x_total_cov_mat_inverse(self):
        """inverse of the total *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return invert_matrix(self.x_total_cov_mat)

    @property
    def y_total_cov_mat_inverse(self):
        """inverse of the total *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return invert_matrix(self.y_total_cov_mat)

    @property
    def projected_xy_total_cov_mat_inverse(self):
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return invert_matrix(self.projected_xy_total_cov_mat)

    @property
    def y_total_uncor_cov_mat(self):
        """the total *y* uncorrelated covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self.y_data_uncor_cov_mat + self.y_model_uncor_cov_mat

    @property
    def y_total_uncor_cov_mat_inverse(self):
        """inverse of the uncorrelated part of the total *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return invert_matrix(self.y_total_uncor_cov_mat)

    @property
    def _y_total_nuisance_cor_design_mat(self):
        """matrix containing the correlated parts of all model uncertainties for all total points"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self._y_data_nuisance_cor_design_mat


    @property
    def x_total_uncor_cov_mat(self):
        """the total *x* uncorrelated covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self.x_total_uncor_cov_mat

    @property
    def x_total_uncor_cov_mat_inverse(self):
        """inverse of the total *x* uncorrelated covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return invert_matrix(self.x_total_uncor_cov_mat)

    @property
    def y_error_band(self):
        """one-dimensional array representing the uncertainty band around the model function"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        return self._calculate_y_error_band()

    def get_y_error_band(self, index):
        #TODO documentation
        #TODO num_points config
        num_points = 100
        return self.y_error_band[index * num_points : (index + 1) * num_points]

    @property
    def x_range(self):
        """range of the *x* measurement data"""
        return self._data_container.x_range

    def get_x_range(self, index):
        """range of the *x* measurement data for one spcific model"""
        return self._data_container.get_x_range(index)

    @property
    def y_range(self):
        """range of the *y* measurement data"""
        return self._data_container.y_range

    @property
    def poi_values(self):
        # gives the values of the model_function_parameters
        _poi_values = []
        for _name in self._poi_names:
            _poi_values.append(self.parameter_name_value_dict[_name])
        return np.array(_poi_values)

    @property
    def x_uncor_nuisance_values(self):
        """gives the x uncorrelated nuisance vector"""
        _values = []
        for _name in self._x_uncor_nuisance_names:
            _values.append(self.parameter_name_value_dict[_name])
        return np.asarray(_values)

    @property
    def model_count(self):
        """the number of model functions contained in the fit"""
        return self._model_function.model_function_count

    # -- public methods

    def add_simple_error(self, axis, err_val,
                         model_index=None, name=None, correlation=0, relative=False, reference='data'):
        """
        Add a simple uncertainty source for axis to the data container.
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
        #TODO update documentation
        #TODO None good default value for model_index?
        _ret = super(XYMultiFit, self).add_simple_error(err_val=err_val,
                                                   name=name,
                                                   model_index=model_index,
                                                   correlation=correlation,
                                                   relative=relative,
                                                   reference=reference,
                                                   axis=axis)

        # need to reinitialize the nexus, since simple errors are
        # possibly relevant for nuisance parameters
        self._init_nexus()
        # initialize the Fitter
        self._initialize_fitter()

        return _ret

    def add_simple_shared_x_error(self, err_val, name=None, correlation=0, relative=False, reference='data'):
        """
        Add a simple uncertainty source for the *x*-axis to all datasets or models.
        The correlation between different datasets/models is set to 1.0.
        This method is designed for datasets which contain the same *x*-data.
        Returns an error name which uniquely identifies the created error source.

        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :param reference: which reference values to use when calculating absolute errors from relative errors
        :type reference: 'data' or 'model'
        :return: error name
        :rtype: str
        """
        #TODO move to container?
        if not self._data_container.all_datasets_same_size:
            raise XYMultiFitException("You can only add a shared x error if all datasets are of the same size!")
        _lower, _upper = self._data_container.get_data_bounds(0)
        _x_size = _upper - _lower
        try:
            err_val.ndim   # will raise if simple float
        except AttributeError:
            err_val = np.asarray(err_val, dtype=float)

        if err_val.ndim == 0:  # if dimensionless numpy array (i.e. float64), add a dimension
            err_val = np.ones(_x_size) * err_val
        elif err_val.size != _x_size:
            raise XYMultiFitException("Expected to receive %s error values but received %s instead!" %
                                    (_x_size, err_val.size))
        _total_err_val = np.tile(err_val, self._data_container.num_datasets)
        _correlation_matrix = np.maximum(np.eye(_x_size), np.ones((_x_size, _x_size)) * correlation)
        _total_correlation_matrix = np.tile(_correlation_matrix,
                                            (self._data_container.num_datasets, self._data_container.num_datasets))
        return self.add_matrix_error('x', err_matrix=_total_correlation_matrix, matrix_type='cor',
                                     name=name, err_val=_total_err_val, relative=relative, reference=reference)


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
        _ret = super(XYMultiFit, self).add_matrix_error(err_matrix=err_matrix,
                                                   matrix_type=matrix_type,
                                                   name=name,
                                                   err_val=err_val,
                                                   relative=relative,
                                                   reference=reference,
                                                   axis=axis)

        # do not reinitialize the nexus, since matrix errors are not
        # relevant for nuisance parameters

        return _ret

    def set_poi_values(self, param_values):
        """set the start values of all parameters of interests"""
        _param_names = self._poi_names
        #test list length
        if not len(param_values) == len(_param_names):
            raise XYMultiFitException("Cannot set all fit parameter values: %d fit parameters declared, "
                                       "but %d provided!"
                                       % (len(_param_names), len(param_values)))
        # set values in nexus
        _par_val_dict = {_pn: _pv for _pn, _pv in zip(_param_names, param_values)}
        self.set_parameter_values(**_par_val_dict)

    def do_fit(self):
        """Perform the fit."""
        if self._cost_function.needs_errors and not self._data_container.has_y_errors:
            self._cost_function.on_no_errors()
        if not self._data_container.has_x_errors:
            super(XYMultiFit, self).do_fit()
        else:
            self._fitter.do_fit()
            _convergence_limit = float(kc('fit', 'x_error_fit_convergence_limit'))
            _previous_cost_function_value = self.cost_function_value
            for i in range(kc('fit', 'max_x_error_fit_iterations')):
                self._fitter.do_fit()
                if np.abs(self.cost_function_value - _previous_cost_function_value) < _convergence_limit:
                    break
                _previous_cost_function_value = self.cost_function_value
            self._loaded_result_dict = None
            self._update_parameter_formatters()

    def assign_model_function_expression(self, expression_format_string, model_index):
        """Assign a plain-text-formatted expression string to the model function."""
        self._model_function.assign_model_function_expression(expression_format_string, model_index)

    def assign_model_function_latex_expression(self, latex_expression_format_string, model_index):
        """Assign a LaTeX-formatted expression string to the model function."""
        self._model_function.assign_model_function_latex_expression(latex_expression_format_string, model_index)

    def eval_model_function(self, x=None, model_parameters=None, model_index=None):
        """
        Evaluate the model function.

        :param x: values of *x* at which to evaluate the model function (if ``None``, the data *x* values are used)
        :type x: iterable of float
        :param model_parameters: the model parameter values (if ``None``, the current values are used)
        :type model_parameters: iterable of float
        :param model_index: the index of the model function to be evaluated
        :type model_index: int
        :return: model function values
        :rtype: :py:class:`numpy.ndarray`
        """
        #TODO update documentation
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.eval_model_function(
            x=x,
            model_parameters=model_parameters,
            model_index=model_index
        )

    def calculate_nuisance_parameters(self):
        """
        Calculate and return the nuisance parameter values.

        NOTE: currently only works for calculating nuisance parameters
        for correlated 'y' uncertainties.

        :return: vector containing the nuisance parameter values
        :rtype: ``numpy.array``
        """
        _uncor_cov_mat_inverse = self.y_data_uncor_cov_mat_inverse
        _cor_cov_mat = self._y_data_nuisance_cor_design_mat
        _y_data = self.y_data
        _y_model = self.eval_model_function(x=self.x_model)

        # retrieve the errors for which to assign nuisance parameters
        _nuisance_error_objects = self.get_matching_errors(
            matching_criteria=dict(
                axis=1,  # cannot use 'y' here
                correlated=True
            )
        )

        _nuisance_size = len(_nuisance_error_objects)
        _residuals = _y_data - _y_model

        _left_side = (_cor_cov_mat).dot(_uncor_cov_mat_inverse).dot(np.transpose(_cor_cov_mat))
        _left_side += np.eye(_nuisance_size, _nuisance_size)
        _right_side = np.asarray(_cor_cov_mat).dot(np.asarray(_uncor_cov_mat_inverse)).dot(_residuals)

        _nuisance_vector = np.linalg.solve(_left_side, np.transpose(_right_side))

        return _nuisance_vector

    def get_splice(self, data, index):
        """
        Utility function to splice an iterable according to the container data indices.
        Specifically, this method splices consecutive data for all models to just the data
        of the model with the given index.
        :param data: the data to be spliced
        :type data: iterable
        :param index: the index of the model whose data will be spliced out
        :type int:
        :return the spliced data
        """
        return self._data_container.get_splice(data, index)

    def get_model_function(self, index):
        """
        Returns the specified native Phython function which is used as a model function.
        :param index: the index of the model function to be returned
        :type int:
        :return the model function with the specified index
        """
        return self._model_function.get_underlying_model_function(index)
