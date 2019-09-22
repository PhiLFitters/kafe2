from collections import OrderedDict
from copy import deepcopy

import six
import numpy as np

from ...config import kc
from ...core import NexusFitter, Nexus
from .._base import (FitException, FitBase, DataContainerBase,
                     ModelParameterFormatter, CostFunctionBase)
from .container import HistContainer
from .cost import HistCostFunction_NegLogLikelihood, HistCostFunction_UserDefined, STRING_TO_COST_FUNCTION
from .model import HistParametricModel, HistModelFunction
from ..util import function_library

__all__ = ["HistFit"]


class HistFitException(FitException):
    pass


class HistFit(FitBase):
    CONTAINER_TYPE = HistContainer
    MODEL_TYPE = HistParametricModel
    MODEL_FUNCTION_TYPE = HistModelFunction
    EXCEPTION_TYPE = HistFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'model_density', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}

    def __init__(self, data, model_density_function=function_library.normal_distribution_pdf,
                 cost_function=HistCostFunction_NegLogLikelihood(data_point_distribution='poisson'),
                 model_density_antiderivative=None, minimizer=None, minimizer_kwargs=None):
        """
        Construct a fit of a model to a histogram.

        :param data: the measurement values
        :type data: iterable of float
        :param model_density_function: the model density function
        :type model_density_function: :py:class:`~kafe2.fit.hist.HistModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        """
        # set/construct the model function object
        if isinstance(model_density_function, self.__class__.MODEL_FUNCTION_TYPE):
            # TODO shouldn't this Exception only be raised if the kafe2 model function already has an antiderivative?
            if model_density_antiderivative is not None:
                raise HistFitException("Antiderivative (%r) provided in constructor for %r, "
                                       "but histogram model function object (%r) already constructed!"
                                       % (model_density_antiderivative, self.__class__, model_density_function))
            self._model_function = model_density_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_density_function, model_density_antiderivative=model_density_antiderivative)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        # set and validate the cost function
        if isinstance(cost_function, CostFunctionBase):
            self._cost_function = cost_function
        elif isinstance(cost_function, str):
            self._cost_function = STRING_TO_COST_FUNCTION[cost_function]()
        else:
            self._cost_function = HistCostFunction_UserDefined(cost_function)
            # self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?

        # declare cache
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None

        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._initialize_fitter(minimizer, minimizer_kwargs)

        self._fit_param_constraints = []
        self._loaded_result_dict = None

        # set the data after the cost_function has been set and nexus has been initialized
        self.data = data

    # -- private methods

    def _init_nexus(self):
        self._nexus = Nexus()
        self._nexus.new_function(lambda: self.data, function_name='data')  # Node containing indexed data is called 'data'

        # get names and default values of all parameters
        _nexus_new_dict = self._get_default_values(model_function=self._model_function,
                                                   x_name=self._model_function.x_name)
        self._nexus.new(**_nexus_new_dict)  # Create nexus Nodes for function parameters

        self._fit_param_names = []  # names of all fit parameters (including nuisance parameters)
        for param_name in six.iterkeys(_nexus_new_dict):
            self._fit_param_names.append(param_name)

        #self._nexus.new_function(self._model_func_handle, add_unknown_parameters=False)

        # add an alias 'model' for accessing the model values
        #self._nexus.new_alias(**{'model': self._model_func_handle.__name__})
        #self._nexus.new_alias(**{'model_density': self._model_func_handle.__name__})

        # bind 'model' node
        self._nexus.new_function(lambda: self.model, function_name='model')
        # need to set dependencies manually
        for _fpn in self._fit_param_names:
            self._nexus.add_dependency(_fpn, 'model')
        # bind other reserved nodes
        self._nexus.new_function(lambda: self.data_error, function_name='data_error')
        self._nexus.new_function(lambda: self.data_cov_mat, function_name='data_cov_mat')
        self._nexus.new_function(lambda: self.data_cov_mat_inverse, function_name='data_cov_mat_inverse')
        self._nexus.new_function(lambda: self.model_error, function_name='model_error')
        self._nexus.new_function(lambda: self.model_cov_mat, function_name='model_cov_mat')
        self._nexus.new_function(lambda: self.model_cov_mat, function_name='model_cov_mat_inverse')
        self._nexus.new_function(lambda: self.total_error, function_name='total_error')
        self._nexus.new_function(lambda: self.total_cov_mat, function_name='total_cov_mat')
        self._nexus.new_function(lambda: self.total_cov_mat_inverse, function_name='total_cov_mat_inverse')
        self._nexus.new_function(lambda: self.parameter_values, function_name='parameter_values')
        self._nexus.new_function(lambda: self.parameter_constraints, function_name='parameter_constraints')

        # the cost function (the function to be minimized)
        self._nexus.new_function(self._cost_function.func, function_name=self._cost_function.name, add_unknown_parameters=False)
        self._nexus.new_alias(**{'cost': self._cost_function.name})

        for _arg_name in self._fit_param_names:
            self._nexus.add_dependency(source=_arg_name, target="parameter_values")

    def _invalidate_total_error_cache(self):
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None

    def _mark_errors_for_update(self):
        # TODO: implement a mass 'mark_for_update' routine in Nexus
        self._nexus.get_by_name('model').mark_for_update()
        self._nexus.get_by_name('data_error').mark_for_update()
        self._nexus.get_by_name('data_cov_mat').mark_for_update()
        self._nexus.get_by_name('data_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('model_error').mark_for_update()
        self._nexus.get_by_name('model_cov_mat').mark_for_update()
        self._nexus.get_by_name('model_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('total_error').mark_for_update()
        self._nexus.get_by_name('total_cov_mat').mark_for_update()
        self._nexus.get_by_name('total_cov_mat_inverse').mark_for_update()

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise HistFitException("Incompatible container type '{}' (expected '{}')"
                                   .format(type(new_data), self.CONTAINER_TYPE))
        else:
            raise HistFitException("Fitting a histogram requires a HistContainer!")
        self._nexus.get_by_name('data').mark_for_update()

    def _set_new_parametric_model(self):
        # create the child ParametricModel object
        self._param_model = self._new_parametric_model(self._data_container.size,
                                                       self._data_container.bin_range,
                                                       self._model_function,
                                                       self.parameter_values,
                                                       self._data_container.bin_edges,
                                                       model_density_func_antiderivative=
                                                       self._model_function.antiderivative)
        self._mark_errors_for_update_invalidate_total_error_cache()

    # -- public properties

    @FitBase.data.getter
    def data(self):
        """array of measurement values"""
        return self._data_container.data

    @property
    def data_error(self):
        """array of pointwise data uncertainties"""
        return self._data_container.err

    @property
    def data_cov_mat(self):
        """the data covariance matrix"""
        return self._data_container.cov_mat

    @property
    def data_cov_mat_inverse(self):
        """inverse of the data covariance matrix (or ``None`` if singular)"""
        return self._data_container.cov_mat_inverse

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data * self._data_container.n_entries  # NOTE: model is just a density->scale up

    @property
    def model_error(self):
        """array of pointwise model uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.err  # FIXME: how to handle scaling

    @property
    def model_cov_mat(self):
        """the model covariance matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cov_mat

    @property
    def model_cov_mat_inverse(self):
        """inverse of the model covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cov_mat_inverse

    @property
    def total_error(self):
        """array of pointwise total uncertainties"""
        if self.__cache_total_error is None:
            _tmp = self.data_error**2
            _tmp += self.model_error**2
            self.__cache_total_error = np.sqrt(_tmp)
        return self.__cache_total_error

    @property
    def total_cov_mat(self):
        """the total covariance matrix"""
        if self.__cache_total_cov_mat is None:
            _tmp = self.data_cov_mat
            _tmp += self.model_cov_mat
            self.__cache_total_cov_mat = _tmp
        return self.__cache_total_cov_mat

    @property
    def total_cov_mat_inverse(self):
        """inverse of the total covariance matrix (or ``None`` if singular)"""
        if self.__cache_total_cov_mat_inverse is None:
            _tmp = self.total_cov_mat
            try:
                _tmp = np.linalg.inv(_tmp)
                self.__cache_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_total_cov_mat_inverse

    # -- public methods

    ## add_error... methods inherited from FitBase ##

    def eval_model_function_density(self, x, model_parameters=None):
        """
        Evaluate the model function density.

        :param x: values of *x* at which to evaluate the model function density
        :type x: iterable of float
        :param model_parameters: the model parameter values (if ``None``, the current values are used)
        :type model_parameters: iterable of float
        :return: model function density values
        :rtype: :py:class:`numpy.ndarray`
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.eval_model_function_density(x=x, model_parameters=model_parameters)
