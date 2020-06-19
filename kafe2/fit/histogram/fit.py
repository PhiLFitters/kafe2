from collections import OrderedDict
from copy import deepcopy

import six
import numpy as np

from ...config import kc
from ...core import NexusFitter, Nexus
from ...core.fitters.nexus import Parameter, Alias, NexusError
from .._base import FitException, FitBase, DataContainerBase
from .container import HistContainer
from .._base.cost import CostFunction_NegLogLikelihood, CostFunction, STRING_TO_COST_FUNCTION
from .model import HistParametricModel, HistModelFunction
from .plot import HistPlotAdapter
from ..util import function_library, add_in_quadrature, collect, invert_matrix

__all__ = ['HistFit', 'HistFitException']


class HistFitException(FitException):
    pass


class HistFit(FitBase):
    CONTAINER_TYPE = HistContainer
    MODEL_TYPE = HistParametricModel
    MODEL_FUNCTION_TYPE = HistModelFunction
    PLOT_ADAPTER_TYPE = HistPlotAdapter
    EXCEPTION_TYPE = HistFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'model_density', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}
    _BASIC_ERROR_NAMES = {'data_error', 'model_error', 'data_cov_mat', 'model_cov_mat'}

    def __init__(self,
                 data,
                 model_density_function=function_library.normal_distribution_pdf,
                 cost_function=CostFunction_NegLogLikelihood(
                    data_point_distribution='poisson'),
                 bin_evaluation="simpson",
                 minimizer=None,
                 minimizer_kwargs=None):
        """
        Construct a fit of a model to a histogram. If bin_evaluation is a Python function or
        of a numpy.vectorize object it is interpreted as the antiderivative of
        model_density_function. If bin_evaluation is equal to "rectangle", "midpoint", "trapezoid",
        or "simpson" the bin heights are evaluated according to the corresponding quadrature
        formula. If bin_evaluation is equal to "numerical" the bin heights are evaluated by
        numerically integrating model_density_function.

        :param data: a :py:class:`~kafe2.fit.hist.HistContainer` representing histogrammed data
        :type data: :py:class:`~kafe2.fit.hist.HistContainer`
        :param model_density_function: the model density function
        :type model_density_function: :py:class:`~kafe2.fit.hist.HistModelFunction` or unwrapped
        native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped
        native Python function
        :param bin_evaluation: how the model evaluates bin heights.
        :type bin_evaluation: str, callable, or numpy.vectorize
        :param minimizer: the minimizer to use for fitting.
        :type minimizer: None, "iminuit", "tminuit", or "scipy".
        :param minimizer_kwargs: dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """
        # set/construct the model function object
        if isinstance(model_density_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_density_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_density_function)
        self._bin_evaluation = bin_evaluation

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        # set and validate the cost function
        if isinstance(cost_function, CostFunction):
            self._cost_function = cost_function
        elif isinstance(cost_function, str):
            try:
                _cost_function_class, _kwargs = STRING_TO_COST_FUNCTION[cost_function]
            except KeyError:
                raise HistFitException('Unknown cost function: %s' % cost_function)
            self._cost_function = _cost_function_class(**_kwargs)
        else:
            self._cost_function = CostFunction(cost_function)
            # self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?

        # initialize the Nexus
        self._init_nexus()

        # save minimizer, minimizer_kwargs for serialization
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs

        # initialize the Fitter
        self._initialize_fitter()

        self._fit_param_constraints = []
        self._loaded_result_dict = None

        # set the data after the cost_function has been set and nexus has been initialized
        self.data = data

    # -- private methods

    def _init_nexus(self):

        self._nexus = Nexus()

        for _type in ('data', 'model'):

            # add data and model for axis
            self._add_property_to_nexus(_type)
            # add errors for axis
            self._add_property_to_nexus('_'.join((_type, 'error')))

            # add cov mats for axis
            for _prop in ('cov_mat',):  # TODO: 'uncor_cov_mat'
                _node = self._add_property_to_nexus('_'.join((_type, _prop)))
                # add inverse
                self._nexus.add_function(
                    invert_matrix,
                    func_name='_'.join((_node.name, 'inverse')),
                    par_names=(_node.name,)
                )

        # 'total_error', i.e. data + model error in quadrature
        self._nexus.add_function(
            add_in_quadrature,
            func_name='total_error',
            par_names=(
                'data_error',
                'model_error'
            )
        )

        # 'total_cov_mat', i.e. data + model cov mats
        for _mat in ('cov_mat',):  # TODO: 'uncor_cov_mat'
            _node = (
                self._nexus.get('_'.join(('data', _mat))) +
                self._nexus.get('_'.join(('model', _mat)))
            )
            _node.name = '_'.join(('total', _mat))
            self._nexus.add(_node)

            # add inverse
            self._nexus.add_function(
                invert_matrix,
                func_name='_'.join((_node.name, 'inverse')),
                par_names=(_node.name,),
            )

        # get names and default values of all parameters
        _nexus_new_dict = self._get_default_values(
            model_function=self._model_function,
            x_name=self._model_function.x_name
        )

        # -- fit parameters

        self._fit_param_names = []  # names of all fit parameters (including nuisance parameters)
        self._poi_names = []  # names of the parameters of interest (i.e. the model parameters)
        for _par_name, _par_value in six.iteritems(_nexus_new_dict):
            # create nexus node for function parameter
            self._nexus.add(Parameter(_par_value, name=_par_name))

            # keep track of fit parameter names
            self._fit_param_names.append(_par_name)
            self._poi_names.append(_par_name)

        # -- nuisance parameters
        self._nuisance_names = []  # names of all nuisance parameters accounting for correlated errors

        self._nexus.add_function(lambda: self.poi_values, func_name='poi_values')
        self._nexus.add_function(lambda: self.parameter_values, func_name='parameter_values')
        self._nexus.add_function(lambda: self.parameter_constraints, func_name='parameter_constraints')

        # add the original function name as an alias to 'model'
        try:
            self._nexus.add_alias(self._model_function.name, alias_for='model')
        except NexusError:
            pass  # allow 'model' as function name for model

        self._nexus.add_function(
            collect,
            func_name="nuisance_vector"
        )

        # -- initialize nuisance parameters

        # TODO: implement nuisance parameters for histograms

        # the cost function (the function to be minimized)
        _cost_node = self._nexus.add_function(
            self._cost_function,
            func_name=self._cost_function.name,
            par_names=self._cost_function.arg_names
        )

        _cost_alias = self._nexus.add_alias('cost', alias_for=self._cost_function.name)

        self._nexus.add_dependency('poi_values', depends_on=self._poi_names)
        self._nexus.add_dependency('parameter_values', depends_on=self._fit_param_names)

        self._nexus.add_dependency(
            'model',
            depends_on=(
                'poi_values'
            )
        )

        self._nexus.add_dependency(
            'total_error',
            depends_on=(
                'data_error',
                'model_error',
            )
        )

        # TODO: add 'uncor_cov_mat'
        #for _side in ('data', 'model', 'total'):
        #    self._nexus.add_dependency(
        #        '{}_uncor_cov_mat'.format(_side),
        #        depends_on='{}_cov_mat'.format(_side)
        #    )

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise HistFitException("Incompatible container type '{}' (expected '{}')"
                                   .format(type(new_data), self.CONTAINER_TYPE))
        else:
            raise HistFitException("Fitting a histogram requires a HistContainer!")
        self._data_container._on_error_change_callback = self._on_error_change

        self._nexus.get('data').mark_for_update()

    def _set_new_parametric_model(self):
        # create the child ParametricModel object
        self._param_model = self._new_parametric_model(
            self._data_container.size, self._data_container.bin_range, self._model_function,
            self.parameter_values, self._data_container.bin_edges,
            bin_evaluation=self._bin_evaluation)
        self._param_model._on_error_change_callbacks = [self._on_error_change]

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
        return add_in_quadrature(self.data_error, self.model_error)

    @property
    def total_cov_mat(self):
        """the total covariance matrix"""
        return self.data_cov_mat + self.model_cov_mat

    @property
    def total_cov_mat_inverse(self):
        """inverse of the total covariance matrix (or ``None`` if singular)"""
        return invert_matrix(self.total_cov_mat)

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
