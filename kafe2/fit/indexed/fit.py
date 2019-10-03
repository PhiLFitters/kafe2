from collections import OrderedDict
from copy import deepcopy

import numpy as np
import sys
import textwrap

from ...tools import print_dict_as_table
from ...config import kc
from ...core import NexusFitter, Nexus
from .._base import FitException, FitBase, DataContainerBase, CostFunctionBase
from .container import IndexedContainer
from .cost import IndexedCostFunction_Chi2, IndexedCostFunction_UserDefined, STRING_TO_COST_FUNCTION
from .model import IndexedParametricModel, IndexedModelFunction


__all__ = ["IndexedFit"]


class IndexedFitException(FitException):
    pass


class IndexedFit(FitBase):
    CONTAINER_TYPE = IndexedContainer
    MODEL_TYPE = IndexedParametricModel
    MODEL_FUNCTION_TYPE = IndexedModelFunction
    EXCEPTION_TYPE = IndexedFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}

    def __init__(self, data, model_function, cost_function=IndexedCostFunction_Chi2(errors_to_use='covariance', fallback_on_singular=True), minimizer=None, minimizer_kwargs=None):
        """
        Construct a fit of a model to a series of indexed measurements.

        :param data: the measurement values
        :type data: iterable of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe2.fit.indexed.IndexedModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        """
        # set the data
        self.data = data

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
        elif isinstance(cost_function, str):
            self._cost_function = STRING_TO_COST_FUNCTION[cost_function]()
        else:
            self._cost_function = IndexedCostFunction_UserDefined(cost_function)
            #self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?
        _data_and_cost_compatible, _reason = self._cost_function.is_data_compatible(self.data)
        if not _data_and_cost_compatible:
            raise self.EXCEPTION_TYPE('Fit data and cost function are not compatible: %s' % _reason)

        # declare cache
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None

        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._initialize_fitter(minimizer, minimizer_kwargs)
        # create the child ParametricModel objet
        self._param_model = self._new_parametric_model(self._model_function, self.parameter_values, shape_like=self.data)

        # TODO: check where to update this (set/release/etc.)
        # FIXME: nicer way than len()?
        self._cost_function.ndf = self._data_container.size - len(self._param_model.parameters)

        self._fit_param_constraints = []
        self._loaded_result_dict = None


    # -- private methods

    def _init_nexus(self):
        self._nexus = Nexus()
        self._nexus.new_function(lambda: self.data, function_name='data')  # Node containing indexed data is called 'data'

        # create a NexusNode for each parameter of the model function

        _nexus_new_dict = OrderedDict()
        _arg_defaults = self._model_function.argspec.defaults
        _n_arg_defaults = 0 if _arg_defaults is None else len(_arg_defaults)
        self._fit_param_names = []
        for _arg_pos, _arg_name in enumerate(self._model_function.argspec.args):
            if _arg_pos >= (self._model_function.argcount - _n_arg_defaults):
                _default_value = _arg_defaults[_arg_pos - (self._model_function.argcount - _n_arg_defaults)]
            else:
                _default_value = kc('core', 'default_initial_parameter_value')
            _nexus_new_dict[_arg_name] = _default_value
            self._fit_param_names.append(_arg_name)

        self._nexus.new(**_nexus_new_dict)  # Create nexus Nodes for function parameters

        self._nexus.new_function(self._model_function.func, function_name=self._model_function.name, add_unknown_parameters=False)

        # add an alias 'model' for accessing the model values
        self._nexus.new_alias(**{'model': self._model_function.name})

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
        self._nexus.get_by_name('data_error').mark_for_update()
        self._nexus.get_by_name('data_cov_mat').mark_for_update()
        self._nexus.get_by_name('data_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('model_error').mark_for_update()
        self._nexus.get_by_name('model_cov_mat').mark_for_update()
        self._nexus.get_by_name('model_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('total_error').mark_for_update()
        self._nexus.get_by_name('total_cov_mat').mark_for_update()
        self._nexus.get_by_name('total_cov_mat_inverse').mark_for_update()

    # -- public properties

    @property
    def data(self):
        """array of measurement values"""
        return self._data_container.data

    @data.setter
    def data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise IndexedFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = self._new_data_container(new_data, dtype=float)
        # Fixme: Error when performing fit if data has a different shape than the old data. model still has old shape
        #        Updating the data node is not enough. The parametric model needs to be rebuilt as well.
        #        When rebuilding the parametric model, the shape of the model function needs to be changed too
        if hasattr(self, '_nexus'):  # update nexus data nodes
            self._nexus.get_by_name('data').mark_for_update()
        if hasattr(self, '_cost_function'):
            self._cost_function.ndf = self._data_container.size - len(self._param_model.parameters)

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
    def data_cor_mat(self):
        """the data correlation matrix"""
        return self._data_container.cor_mat

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data

    @property
    def model_error(self):
        """array of pointwise model uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.err

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
    def model_cor_mat(self):
        """the model correlation matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cor_mat

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
                _tmp = _tmp.I
                self.__cache_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_total_cov_mat_inverse

    # -- public methods

    ## add_error... methods inherited from FitBase ##

    def generate_plot(self):
        from kafe2.fit.indexed import IndexedPlot
        return IndexedPlot(self)

    def report(self, output_stream=sys.stdout,
               show_data=True,
               show_model=True,
               asymmetric_parameter_errors=False):
        """
        Print a summary of the fit state and/or results.

        :param output_stream: the output stream to which the report should be printed
        :type output_stream: TextIOBase
        :param show_data: if ``True``, print out information about the data
        :type show_data: bool
        :param show_model: if ``True``, print out information about the parametric model
        :type show_model: bool
        :param asymmetric_parameter_errors: if ``True``, use two different parameter errors for up/down directions
        :type asymmetric_parameter_errors: bool
        """
        _indent = ' ' * 4

        if show_data:
            output_stream.write(textwrap.dedent("""
                ########
                # Data #
                ########

            """))

            _data_table_dict = OrderedDict()
            _data_table_dict['Index'] = range(self.data_size)
            _data_table_dict['Data'] = self.data
            if self.has_data_errors:
                _data_table_dict['Data Error'] = self.data_error
                #_data_table_dict['Data Total Covariance Matrix'] = self.data_cov_mat
                _data_table_dict['Data Total Correlation Matrix'] = self.data_cor_mat

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

            _data_table_dict = OrderedDict()
            _data_table_dict['Index'] = range(self.data_size)
            _data_table_dict['Model'] = self.model
            if self.has_model_errors:
                _data_table_dict['Model Error'] = self.model_error
                #_data_table_dict['Model Total Covariance Matrix'] = self.model_cov_mat
                _data_table_dict['Model Total Correlation Matrix'] = self.model_cor_mat

            print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=1)

        super(IndexedFit, self).report(output_stream=output_stream,
                                       asymmetric_parameter_errors=asymmetric_parameter_errors)
