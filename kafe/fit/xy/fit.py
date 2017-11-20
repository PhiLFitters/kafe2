from collections import OrderedDict
from copy import deepcopy

import numpy as np
import sys
import textwrap

from ...tools import print_dict_as_table
from ...core import NexusFitter, Nexus
from ...core.error import MatrixGaussianError, SimpleGaussianError
from ...config import kc
from .._base import (FitException, FitBase, DataContainerBase,
                     ModelParameterFormatter, CostFunctionBase)
from .container import XYContainer
from .cost import XYCostFunction_Chi2, XYCostFunction_UserDefined
from .format import XYModelFunctionFormatter
from .model import XYParametricModel, XYModelFunction


__all__ = ["XYFit"]


class XYFitException(FitException):
    pass


class XYFit(FitBase):
    CONTAINER_TYPE = XYContainer
    MODEL_TYPE = XYParametricModel
    MODEL_FUNCTION_TYPE = XYModelFunction
    EXCEPTION_TYPE = XYFitException
    RESERVED_NODE_NAMES = {'y_data', 'y_model', 'cost',
                           'x_error', 'y_data_error', 'y_model_error', 'total_error',
                           'x_cov_mat', 'y_data_cov_mat', 'y_model_cov_mat', 'total_cov_mat',
                           'x_cor_mat', 'y_data_cor_mat', 'y_model_cor_mat', 'total_cor_mat',
                           'x_cov_mat_inverse', 'y_data_cov_mat_inverse', 'y_model_cov_mat_inverse', 'total_cor_mat_inverse'
                           'y_data_uncor_cov_mat', 'y_model_uncor_cov_mat','y_total_uncor_cov_mat',
                           'nuisance_y_data_cor_cov_mat','nuisance_y_model_cor_cov_mat','nuisance_y_total_cor_cov_mat',
                           'nuisance_para', 'y_nuisance_vector'}

    def __init__(self, xy_data, model_function,
                 cost_function=XYCostFunction_Chi2(axes_to_use='xy', errors_to_use='covariance'),
                 minimizer=None, minimizer_kwargs=None):
        """
        Construct a fit of a model to *xy* data.

        :param xy_data: the x and y measurement values
        :type xy_data: (2, N)-array of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe.fit.indexed.XYModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        """
        # set the data
        self.data = xy_data
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs

        # set/construct the model function object
        if isinstance(model_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_function)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        # set and validate the cost function
        if isinstance(cost_function, CostFunctionBase):
            self._cost_function = cost_function
        else:
            self._cost_function = XYCostFunction_UserDefined(cost_function)
            #self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?

        # declare cache
        self._invalidate_total_error_cache()
        
        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._initialize_fitter(minimizer, minimizer_kwargs)
        # create the child ParametricModel object
        self._param_model = self._new_parametric_model(self.x_model, self._model_function.func,
                                                       self.poi_values)

        # TODO: check where to update this (set/release/etc.)
        # FIXME: nicer way than len()?
        self._cost_function.ndf = self._data_container.size - len(self._param_model.parameters)


    # -- private methods

    def _init_nexus(self):
        self._nexus = Nexus()
        self._nexus.new(y_data=self.y_data)
        self._nexus.new(x_data=self.x_data)

        # create a NexusNode for each parameter of the model function

        _nexus_new_dict = OrderedDict()
        _arg_defaults = self._model_function.argspec.defaults
        _n_arg_defaults = 0 if _arg_defaults is None else len(_arg_defaults)
        self._fit_param_names = [] #every fit parameter name (with nuisance)
        self._func_fit_para_names = [] #just the names of the parameters in the modelfunction
        self._y_nuisance_names = []
        self._x_uncor_nuisance_names = []
       # self._x_cor_nuisance_names = []


        for _arg_pos, _arg_name in enumerate(self._model_function.argspec.args):
            # skip independent variable parameter
            if _arg_name == self._model_function.x_name:
                continue
            if _arg_pos >= (self._model_function.argcount - _n_arg_defaults):
                _default_value = _arg_defaults[_arg_pos - (self._model_function.argcount - _n_arg_defaults)]
            else:
                _default_value = kc('core', 'default_initial_parameter_value')
            _nexus_new_dict[_arg_name] = _default_value
            self._fit_param_names.append(_arg_name)
            self._func_fit_para_names.append(_arg_name)

        self._nexus.new(**_nexus_new_dict)  # Create nexus Nodes for function parameters

        self._nexus.new_function(self._model_function.func, function_name=self._model_function.name,
                                 add_unknown_parameters=False, wire_parameters=False)
        # add an alias 'model' for accessing the model values
        #self._nexus.new_alias(**{'y_model': self._model_function.name})

        # node function for collecting nuisance parameters into a vector
        def _calc_y_nuisance_vector(*n_para):
            return np.asarray(n_para)

        self._nexus.new_function(_calc_y_nuisance_vector, function_name="y_nuisance_vector",
                                 add_unknown_parameters=False)

        # x-nuisance vector for correlated errors
        # def _calc_x_corr_nuisance_vector(*n_para):
        #     return np.asarray(n_para)
        # self._nexus.new_function(_calc_x_corr_nuisance_vector, function_name="x_corr_nuisance_vector",
        #                          add_unknown_parameters=False)

        #x-nuisance vector for uncorrelated errors
        def _calc_x_uncor_nuisance_vector(*n_para):
            return np.asarray(n_para)
        self._nexus.new_function(_calc_x_uncor_nuisance_vector, function_name="x_uncor_nuisance_vector",
                                 add_unknown_parameters=False)

        #initialize nuisance parameters
        if self._cost_function.get_flag("need_y_nuisance") and self._data_container.has_y_errors:
            # y-nuisance parameters
            for i in six.moves.range(self._data_container.y_simple_error_size):
                _eps_name="_eps_{}".format(i)
                self._nexus.new(**{_eps_name:0.0})
                self._nexus.add_dependency(_eps_name, "y_nuisance_vector")
                self._fit_param_names.append(_eps_name)
                self._y_nuisance_names.append(_eps_name)
            self._nexus.set_function_parameter_names("y_nuisance_vector", self._y_nuisance_names)
        if self._cost_function.get_flag("need_x_nuisance") and self._data_container.has_uncor_x_errors:
            # x-nuisance parameters for correlated errors
            # for i in six.moves.range(self._data_container.x_simple_error_size_corelated):
            #     _delta_name = "_delta_cor_{}".format(i)
            #     self._nexus.new(**{_delta_name: 0.0})
            #     self._nexus.add_dependency(_delta_name, "x_cor_nuisance_vector")
            #     self._fit_param_names.append(_delta_name)
            #     _x_cor_nuisance_names.append(_delta_name)
            # self._nexus.set_function_parameter_names("x_cor_nuisance_vector", _x_cor_nuisance_names)
            #x-nuisance paramters for uncorrelated errors
            for i in six.moves.range(self._data_container.size):
                _delta_name = "_delta_un_{}".format(i)
                self._nexus.new(**{_delta_name: 0.0})
                self._nexus.add_dependency(_delta_name, "x_uncor_nuisance_vector")
                self._fit_param_names.append(_delta_name)
                self._x_uncor_nuisance_names.append(_delta_name)
            self._nexus.set_function_parameter_names("x_uncor_nuisance_vector", self._x_uncor_nuisance_names)



        # bind other reserved nodes
        self._nexus.new_function(lambda: self.x_model, function_name='x_model')
        self._nexus.new_function(lambda: self.y_model, function_name='y_model')
        self._nexus.new_function(lambda: self.x_data_error, function_name='x_data_error')
        self._nexus.new_function(lambda: self.x_data_cov_mat, function_name='x_data_cov_mat')
        self._nexus.new_function(lambda: self.x_data_cov_mat_inverse, function_name='x_data_cov_mat_inverse')
        self._nexus.new_function(lambda: self.x_model_error, function_name='x_model_error')
        self._nexus.new_function(lambda: self.x_model_cov_mat, function_name='x_model_cov_mat')
        self._nexus.new_function(lambda: self.x_model_cov_mat_inverse, function_name='x_model_cov_mat_inverse')
        self._nexus.new_function(lambda: self.x_total_error, function_name='x_total_error')
        self._nexus.new_function(lambda: self.x_total_cov_mat, function_name='x_total_cov_mat')
        self._nexus.new_function(lambda: self.x_total_cov_mat_inverse, function_name='x_total_cov_mat_inverse')

        self._nexus.new_function(lambda: self.projected_xy_total_error, function_name='projected_xy_total_error')
        self._nexus.new_function(lambda: self.projected_xy_total_cov_mat, function_name='projected_xy_total_cov_mat')
        self._nexus.new_function(lambda: self.projected_xy_total_cov_mat_inverse, function_name='projected_xy_total_cov_mat_inverse')

        self._nexus.new_function(lambda: self.y_data_error, function_name='y_data_error')
        self._nexus.new_function(lambda: self.y_data_cov_mat, function_name='y_data_cov_mat')
        self._nexus.new_function(lambda: self.y_data_cov_mat_inverse, function_name='y_data_cov_mat_inverse')
        self._nexus.new_function(lambda: self.y_model_error, function_name='y_model_error')
        self._nexus.new_function(lambda: self.y_model_cov_mat, function_name='y_model_cov_mat')
        self._nexus.new_function(lambda: self.y_model_cov_mat_inverse, function_name='y_model_cov_mat_inverse')
        self._nexus.new_function(lambda: self.y_total_error, function_name='y_total_error')
        self._nexus.new_function(lambda: self.y_total_cov_mat, function_name='y_total_cov_mat')
        self._nexus.new_function(lambda: self.y_total_cov_mat_inverse, function_name='y_total_cov_mat_inverse')

        #correlated error matrix (for cor-nuisance approach)
        self._nexus.new_function(lambda: self.nuisance_y_data_cor_cov_mat, function_name='nuisance_y_data_cor_cov_mat')
        self._nexus.new_function(lambda: self.nuisance_y_model_cor_cov_mat, function_name='nuisance_y_model_cor_cov_mat')
        self._nexus.new_function(lambda: self.nuisance_y_total_cor_cov_mat,function_name='nuisance_y_total_cor_cov_mat')

        #uncorrelated y error cov matrix
        self._nexus.new_function(lambda: self.y_data_uncor_cov_mat, function_name='y_data_uncor_cov_mat')
        self._nexus.new_function(lambda: self.y_data_uncor_cov_mat_inverse,function_name='y_data_uncor_cov_mat_inverse')
        self._nexus.new_function(lambda: self.y_model_uncor_cov_mat, function_name='y_model_uncor_cov_mat')
        self._nexus.new_function(lambda: self.y_model_uncor_cov_mat_inverse, function_name='y_model_uncor_cov_mat_inverse')
        self._nexus.new_function(lambda: self.y_total_uncor_cov_mat, function_name='y_total_uncor_cov_mat')
        self._nexus.new_function(lambda: self.y_total_uncor_cov_mat_inverse, function_name='y_total_uncor_cov_mat_inverse')
        # correlated x_error cov matrix (nuisance) TODO: correlated x-errors
        # self._nexus.new_function(lambda: self.nuisance_x_data_cor_cov_mat, function_name='nuisance_x_data_cor_cov_mat')
        # self._nexus.new_function(lambda: self.nuisance_x_model_cor_cov_mat,function_name='nuisance_x_model_cor_cov_mat')
        # self._nexus.new_function(lambda: self.nuisance_x_total_cor_cov_mat,function_name='nuisance_x_total_cor_cov_mat')
        # uncorrelated x error cov matrix
        self._nexus.new_function(lambda: self.x_data_uncor_cov_mat, function_name='x_data_uncor_cov_mat')
        self._nexus.new_function(lambda: self.x_data_uncor_cov_mat_inverse, function_name='x_data_uncor_cov_mat_inverse')
        self._nexus.new_function(lambda: self.x_model_uncor_cov_mat, function_name='x_model_uncor_cov_mat')
        self._nexus.new_function(lambda: self.x_model_uncor_cov_mat_inverse, function_name='x_model_uncor_cov_mat_inverse')
        self._nexus.new_function(lambda: self.x_total_uncor_cov_mat, function_name='x_total_uncor_cov_mat')
        self._nexus.new_function(lambda: self.x_total_uncor_cov_mat_inverse, function_name='x_total_uncor_cov_mat_inverse')

        # the cost function (the function to be minimized)
        self._nexus.new_function(self._cost_function.func, function_name=self._cost_function.name,
                                 add_unknown_parameters=False)

        self._nexus.new_alias(**{'cost': self._cost_function.name})

        # add nexus dependencies to recalculate model
        # whenever nuisance parameters change
        for _arg_name in self._x_uncor_nuisance_names:
             self._nexus.add_dependency(source=_arg_name, target='x_model')

        self._nexus.add_dependency(source='x_data_cov_mat', target='x_model')
        self._nexus.add_dependency(source='x_data_error', target='x_model')

        self._nexus.add_dependency(source='x_model', target="y_model")
        # self._nexus.add_dependency(source='x_uncor_nuisance_vector', target='y_model')
        for _arg_name in self._func_fit_para_names:
            self._nexus.add_dependency(source=_arg_name, target="y_model")



    def _invalidate_total_error_cache(self):
        self.__cache_x_data_error = None
        self.__cache_x_data_cov_mat = None
        self.__cache_x_total_error = None
        self.__cache_x_total_cov_mat = None
        self.__cache_x_total_cov_mat_inverse = None
        self.__cache_projected_xy_total_error = None
        self.__cache_projected_xy_total_cov_mat = None
        self.__cache_projected_xy_total_cov_mat_inverse = None
        self.__cache_y_total_error = None
        self.__cache_y_total_cov_mat = None
        self.__cache_y_total_cov_mat_inverse = None
        self.__cache_y_error_band = None
        self.__cache_y_total_uncor_cov_mat = None
        self.__cache_y_total_uncor_cov_mat_inverse = None
        self.__cache_nuisance_y_total_cor_cov_mat = None
        self.__cache_x_total_uncor_cov_mat = None
        self.__cache_x_total_uncor_cov_mat_inverse = None
        # self.__cache_nuisance_x_total_uncor_cov_mat = None

    def _mark_errors_for_update(self):
        # TODO: implement a mass 'mark_for_update' routine in Nexus
        self._nexus.get_by_name('x_data_error').mark_for_update()
        self._nexus.get_by_name('x_data_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_data_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('x_model_error').mark_for_update()
        self._nexus.get_by_name('x_model_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_model_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('x_total_error').mark_for_update()
        self._nexus.get_by_name('x_total_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_total_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_data_error').mark_for_update()
        self._nexus.get_by_name('y_data_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_data_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_model_error').mark_for_update()
        self._nexus.get_by_name('y_model_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_model_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_total_error').mark_for_update()
        self._nexus.get_by_name('y_total_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_total_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('projected_xy_total_error').mark_for_update()
        self._nexus.get_by_name('projected_xy_total_cov_mat').mark_for_update()
        self._nexus.get_by_name('projected_xy_total_cov_mat_inverse').mark_for_update()

        self._nexus.get_by_name('y_data_uncor_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_model_uncor_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_total_uncor_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_data_uncor_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_model_uncor_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_total_uncor_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('nuisance_y_data_cor_cov_mat').mark_for_update()
        self._nexus.get_by_name('nuisance_y_model_cor_cov_mat').mark_for_update()
        self._nexus.get_by_name('nuisance_y_total_cor_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_data_uncor_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_model_uncor_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_total_uncor_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_data_uncor_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('x_model_uncor_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('x_total_uncor_cov_mat_inverse').mark_for_update()
        # self._nexus.get_by_name('nuisance_x_data_cor_cov_mat').mark_for_update()
        # self._nexus.get_by_name('nuisance_x_model_cor_cov_mat').mark_for_update()
        # self._nexus.get_by_name('nuisance_x_total_cor_cov_mat').mark_for_update()
        # #self._nexus.get_by_name('x_cor_nuisance_vector').mark_for_update() TODO: uncorrelated x-errors
        self._nexus.get_by_name('y_nuisance_vector').mark_for_update()
        self._nexus.get_by_name('x_uncor_nuisance_vector').mark_for_update()
        self._nexus.get_by_name('x_model').mark_for_update()
        self._nexus.get_by_name('y_model').mark_for_update()

    def _mark_errors_for_update_invalidate_total_error_cache(self):
        self._mark_errors_for_update()
        self._invalidate_total_error_cache()

    def _calculate_y_error_band(self):
        _xmin, _xmax = self._data_container.x_range
        _band_x = np.linspace(_xmin, _xmax, 100)  # TODO: config
        _f_deriv_by_params = self._param_model.eval_model_function_derivative_by_parameters(x=_band_x, model_parameters=self.poi_values)
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[par_idx][x_idx]

        _f_deriv_by_params = _f_deriv_by_params.T
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[x_idx][par_idx]

        _band_y = np.zeros_like(_band_x)
        _n_poi = len(self.poi_values)
        for _x_idx, _x_val in enumerate(_band_x):
            _p_res = _f_deriv_by_params[_x_idx]
            _band_y[_x_idx] = _p_res.dot(self.parameter_cov_mat[:_n_poi, :_n_poi]).dot(_p_res)[0, 0]

        self.__cache_y_error_band = np.sqrt(_band_y)

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

    @property
    def data(self):
        """(2, N)-array containing *x* and *y* measurement values"""
        return self._data_container.data

    @data.setter
    def data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise XYFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            _x_data = new_data[0]
            _y_data = new_data[1]
            self._data_container = self._new_data_container(_x_data, _y_data, dtype=float)

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
    def nuisance_y_data_cor_cov_mat(self):
        """matrix containing the correlated parts of all data uncertainties for all data points"""
        return self._data_container.nuisance_y_cor_cov_mat

    @property
    def x_data_uncor_cov_mat(self):
        # data x uncorrelated covariance matrix
        return self._data_container.x_uncor_cov_mat

    @property
    def x_data_uncor_cov_mat_inverse(self):
        # data x uncorrelated inverse covariance matrix
        return self._data_container.x_uncor_cov_mat_inverse

    # @property TODO: correlated x-errors
    # def nuisance_x_data_cor_cov_mat(self):
    #     # date x correlated matrix (nuisance)
    #     return self._data_container.nuisance_y_cor_cov_mat
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
    def nuisance_y_model_cor_cov_mat(self):
        """matrix containing the correlated parts of all model uncertainties for all data points"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.nuisance_y_cor_cov_mat

    @property
    def x_model_cor_mat(self):
        """the model *x* correlation matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y_cor_mat

    # @property TODO: correlated x-errors
    # def nuisance_x_model_cor_cov_mat(self):
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
        if self.__cache_x_total_error is None:
            _tmp = self.x_data_error**2
            _tmp += self.x_model_error**2
            self.__cache_x_total_error = np.sqrt(_tmp)
        return self.__cache_x_total_error

    @property
    def y_total_error(self):
        """array of pointwise total *y* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_y_total_error is None:
            _tmp = self.y_data_error**2
            _tmp += self.y_model_error**2
            self.__cache_y_total_error = np.sqrt(_tmp)
        return self.__cache_y_total_error

    @property
    def projected_xy_total_error(self):
        """array of pointwise total *y* with the x uncertainties projected on top of them"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_error
        if self.__cache_projected_xy_total_error is None:
            _x_errors = self.x_total_error
            _precision = 0.01 * np.min(_x_errors)
            _derivatives = self._param_model.eval_model_function_derivative_by_x(dx=_precision)
            self.__cache_projected_xy_total_error = np.sqrt(self.y_total_error**2 + self.x_total_error**2 * _derivatives**2)
        return self.__cache_projected_xy_total_error

    @property
    def x_total_cov_mat(self):
        """the total *x* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_x_total_cov_mat is None:
            _tmp = self.x_data_cov_mat
            _tmp += self.x_model_cov_mat
            self.__cache_x_total_cov_mat = _tmp
        return self.__cache_x_total_cov_mat

    @property
    def y_total_cov_mat(self):
        """the total *y* covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_y_total_cov_mat is None:
            _tmp = self.y_data_cov_mat
            _tmp += self.y_model_cov_mat
            self.__cache_y_total_cov_mat = _tmp
        return self.__cache_y_total_cov_mat

    @property
    def projected_xy_total_cov_mat(self):
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_cov_mat
        if self.__cache_projected_xy_total_cov_mat is None:
            _x_errors = self.x_total_error
            _precision = 0.01 * np.min(_x_errors)
            _derivatives = self._param_model.eval_model_function_derivative_by_x(dx=_precision)
            _outer_product = np.outer(_derivatives, _derivatives)
            _projected_x_cov_mat = np.asarray(self.x_total_cov_mat) * _outer_product
            self.__cache_projected_xy_total_cov_mat = self.y_total_cov_mat + np.asmatrix(_projected_x_cov_mat)
        return self.__cache_projected_xy_total_cov_mat

    @property
    def x_total_cov_mat_inverse(self):
        """inverse of the total *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_x_total_cov_mat_inverse is None:
            _tmp = self.x_total_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_x_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_x_total_cov_mat_inverse

    @property
    def y_total_cov_mat_inverse(self):
        """inverse of the total *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_y_total_cov_mat_inverse is None:
            _tmp = self.y_total_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_y_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_y_total_cov_mat_inverse

    @property
    def projected_xy_total_cov_mat_inverse(self):
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_projected_xy_total_cov_mat_inverse is None:
            _tmp = self.projected_xy_total_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_projected_xy_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_projected_xy_total_cov_mat_inverse

    @property
    def y_total_uncor_cov_mat(self):
        """the total *y* uncorrelated covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_y_total_uncor_cov_mat is None:
            _tmp = self.y_data_uncor_cov_mat
            _tmp += self.y_model_uncor_cov_mat
            self.__cache_y_total_uncor_cov_mat = _tmp
        return self.__cache_y_total_uncor_cov_mat

    @property
    def y_total_uncor_cov_mat_inverse(self):
        """inverse of the uncorrelated part of the total *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_y_total_uncor_cov_mat_inverse is None:
            _tmp = self.y_total_uncor_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_y_total_uncor_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_y_total_uncor_cov_mat_inverse

    @property
    def nuisance_y_total_cor_cov_mat(self):
        """matrix containing the correlated parts of all model uncertainties for all total points"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_nuisance_y_total_cor_cov_mat is None:
            _tmp = self.nuisance_y_data_cor_cov_mat
            # _tmp += self.nuisance_y_model_cor_cov_mat
            self.__cache_nuisance_y_total_cor_cov_mat = _tmp
        return self.__cache_nuisance_y_total_cor_cov_mat

    @property
    def x_total_uncor_cov_mat(self):
        """the total *x* uncorrelated covariance matrix"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_x_total_uncor_cov_mat is None:
            _tmp = self.x_data_uncor_cov_mat
            _tmp += self.x_model_uncor_cov_mat
            self.__cache_x_total_uncor_cov_mat = _tmp
        return self.__cache_x_total_uncor_cov_mat

    @property
    def x_total_uncor_cov_mat_inverse(self):
        """inverse of the total *x* uncorrelated covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_x_total_uncor_cov_mat_inverse is None:
            _tmp = self.x_total_uncor_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_x_total_uncor_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_x_total_uncor_cov_mat_inverse

    # @property TODO: correlated x-errors
    # def nuisance_x_total_cor_cov_mat(self):
    #     """total *x* correlated covariance matrix (nuisance) (or ``None`` if singular)"""
    #     self._param_model.parameters = self.poi_values  # this is lazy, so just do it
    #     self._param_model.x = self.x_with_errors
    #     if self.__cache_nuisance_x_total_cor_cov_mat is None:
    #         _tmp = self.nuisance_x_data_cor_cov_mat
    #         # _tmp += self.nuisance_x_model_cor_cov_mat
    #         self.__cache_nuisance_x_total_cor_cov_mat = _tmp
    #     return self.__cache_nuisance_x_total_cor_cov_mat

    @property
    def y_error_band(self):
        """one-dimensional array representing the uncertainty band around the model function"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        if self.__cache_y_error_band is None:
            self._calculate_y_error_band()
        return self.__cache_y_error_band

    @property
    def x_range(self):
        """range of the *x* measurement data"""
        return self._data_container.x_range

    @property
    def y_range(self):
        """range of the *y* measurement data"""
        return self._data_container.y_range

    @property
    def poi_values(self):
        # gives the values of the model_function_parameters
        _poi_values = []
        for _name in self._func_fit_para_names:
            _poi_values.append(self.parameter_name_value_dict[_name])
        return _poi_values

    @property
    def x_uncor_nuisance_values(self):
        """gives the x uncorrelated nuisance vector"""
        _values = []
        for _name in self._x_uncor_nuisance_names:
            _values.append(self.parameter_name_value_dict[_name])
        return np.asarray(_values)

    # -- public methods

    def add_simple_error(self, axis, err_val,
                         name=None, correlation=0, relative=False, reference='data'):
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
        _ret = super(XYFit, self).add_simple_error(err_val=err_val,
                                                   name=name,
                                                   correlation=correlation,
                                                   relative=relative,
                                                   reference=reference,
                                                   axis=axis)

        # need to reinitialize the nexus, since simple errors are
        # possibly relevant for nuisance parameters
        self._init_nexus()
        # initialize the Fitter
        self._initialize_fitter(self._minimizer, self._minimizer_kwargs)

        return _ret

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
        _ret = super(XYFit, self).add_matrix_error(err_matrix=err_matrix,
                                                   matrix_type=matrix_type,
                                                   name=name,
                                                   err_val=err_val,
                                                   relative=relative,
                                                   reference=reference,
                                                   axis=axis)

        # do not reinitialize the nexus, since matrix errors are not
        # relevant for nuisance parameters

        return _ret


    def do_fit(self):
        """Perform the fit."""
        if not self._data_container.has_x_errors:
            super(XYFit, self).do_fit()
        else:
            self._fitter.do_fit()
            _convergence_limit = float(kc('fit', 'x_error_fit_convergence_limit'))
            _previous_cost_function_value = self.cost_function_value
            for i in range(kc('fit', 'max_x_error_fit_iterations')):
                self._mark_errors_for_update()
                self._invalidate_total_error_cache()
                self._fitter.do_fit()
                if np.abs(self.cost_function_value - _previous_cost_function_value) < _convergence_limit:
                    break
                _previous_cost_function_value = self.cost_function_value
            # update parameter formatters
            for _fpf, _pv, _pe in zip(self._model_function.argument_formatters, self.parameter_values, self.parameter_errors):
                _fpf.value = _pv
                _fpf.error = _pe

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
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.eval_model_function(x=x, model_parameters=model_parameters)

    # def value_chisquare_nuisance_parameters(self):
    # #calculate nuisance parameters (wrong!!)
    #     for _err_dict in self._data_container._error_dicts.values():
    #         _err = _err_dict["err"]
    #         if isinstance(_err, MatrixGaussianError):
    #             raise XYFitException("Calculating Nuisance Parameters is only "
    #                                        "working with simple errors")
    #
    #          _nuisance_cal = (self.eval_model_function(x=self.x) - self.y_data) / _err.error
    #             _chisquare_uncorr = np.sum((_nuisance_cal * np.sqrt(1.0 - _err._corr_coeff)) ** 2.0)
    #         #_chisquare_corr = (_nuisance[] * np.sqrt(_err._corr_coeff)) ** 2.0
    #
    #         #_chisquare_corr = _chisquare_corr / (self._data_container.size)
    #         _chisquare = _chisquare_uncorr
    #
    #     return _chisquare

    def calculate_nuisance_parameters(self):
        """calculate y-nuisance-vector"""
        _uncor_cov_mat_inverse = self.y_data_uncor_cov_mat_inverse
        _cor_cov_mat = self.nuisance_y_data_cor_cov_mat
        _full_cov_mat_inverse = self.y_data_cov_mat
        _y_data = self.y_data
        _y_model = self.eval_model_function(x=self.x_model)
        _nuisance_size = self._data_container.y_simple_error_size
        _residuals = _y_data - _y_model

        _left_side = (_cor_cov_mat).dot(_uncor_cov_mat_inverse).dot(np.transpose(_cor_cov_mat))+np.eye(_nuisance_size, _nuisance_size)
        _right_side =np.asarray(_cor_cov_mat).dot(np.asarray(_uncor_cov_mat_inverse)).dot(_residuals)

        _nuisance_vector = np.linalg.solve(_left_side, np.transpose(_right_side))


        return _nuisance_vector

    def report(self, output_stream=sys.stdout,
               show_data=True,
               show_model=True):
        """Print a summary of the fit state and/or results."""
        _result_dict = self.get_result_dict()

        _indent = ' ' * 4

        if show_data:
            output_stream.write(textwrap.dedent("""
                ########
                # Data #
                ########

            """))
            _data_table_dict = OrderedDict()
            _data_table_dict['X Data'] = self.x
            if self._data_container.has_x_errors:
                _data_table_dict['X Data Error'] = self.x_error
                #_data_table_dict['X Data Total Covariance Matrix'] = self.x_cov_mat
                _data_table_dict['X Data Total Correlation Matrix'] = self.x_cov_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)
            output_stream.write('\n')

            _data_table_dict = OrderedDict()
            _data_table_dict['Y Data'] = self.y_data
            if self.has_data_errors:
                _data_table_dict['Y Data Error'] = self.y_data_error
                #_data_table_dict['Y Data Total Covariance Matrix'] = self.y_data_cov_mat
                _data_table_dict['Y Data Total Correlation Matrix'] = self.y_data_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)

        if show_model:
            output_stream.write(textwrap.dedent("""
                #########
                # Model #
                #########

            """))

            #output_stream.write(_indent)
            output_stream.write(_indent + "Model Function\n")
            output_stream.write(_indent + "==============\n\n")
            output_stream.write(_indent * 2)
            output_stream.write(
                self._model_function.formatter.get_formatted(
                    with_par_values=False,
                    n_significant_digits=2,
                    format_as_latex=False,
                    with_expression=True
                )
            )
            output_stream.write('\n\n\n')

            # FIXME: x model
            # _data_table_dict = OrderedDict()
            # _data_table_dict['X Model'] = self.x
            # _data_table_dict['X Model Error'] = self.x_error
            # _data_table_dict['X Model Total Covariance Matrix'] = self.x_cov_mat

            # print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)
            # output_stream.write('\n')

            _data_table_dict = OrderedDict()
            _data_table_dict['Y Model'] = self.y_model
            if self.has_model_errors:
                _data_table_dict['Y Model Error'] = self.y_model_error
                #_data_table_dict['Y Model Total Covariance Matrix'] = self.y_model_cov_mat
                _data_table_dict['Y Model Total Correlation Matrix'] = self.y_model_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)

        super(XYFit, self).report()
