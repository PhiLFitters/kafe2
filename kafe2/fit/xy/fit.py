from collections import OrderedDict
from copy import deepcopy

import numpy as np
import six
import sys
import textwrap

from ...tools import print_dict_as_table
from ...core import NexusFitter, Nexus
from ...core.fitters.nexus import Parameter, Alias, NexusError
from ...config import kc
from .._base import FitException, FitBase, DataContainerBase, CostFunction, ModelFunctionBase
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
                           'y_data_uncor_cov_mat', 'y_model_uncor_cov_mat','y_total_uncor_cov_mat',
                           'nuisance_y_data_cor_cov_mat','nuisance_y_model_cor_cov_mat','nuisance_y_total_cor_cov_mat',
                           'nuisance_para', 'y_nuisance_vector',
                           'x_data_cov_mat'}
    _BASIC_ERROR_NAMES = {
        'x_data_error', 'x_model_error', 'x_data_cov_mat', 'x_model_cov_mat',
        'y_data_error', 'y_model_error', 'y_data_cov_mat', 'y_model_cov_mat'
    }

    X_ERROR_ALGORITHMS = ('iterative linear', 'nonlinear')

    def __init__(self,
                 xy_data,
                 model_function=function_library.linear_model,
                 cost_function=XYCostFunction_Chi2(
                    axes_to_use='xy', errors_to_use='covariance'),
                 x_error_algorithm='nonlinear',
                 minimizer=None, minimizer_kwargs=None):
        """
        Construct a fit of a model to *xy* data.

        :param xy_data: the x and y measurement values
        :type xy_data: (2, N)-array of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe2.fit.xy.XYModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        :param x_error_algorithm: algorithm for handling x errors. Can be one of: ``'iterative linear'``, ``'nonlinear'``
        :type x_error_algorithm: str
        :param minimizer: the minimizer to use for fitting.
        :type minimizer: None, "iminuit", "tminuit", or "scipy".
        :param minimizer_kwargs: dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """

        # set/construct the model function object
        if isinstance(model_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_function)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        # set and validate the cost function
        if isinstance(cost_function, CostFunction):
            self._cost_function = cost_function
        elif isinstance(cost_function, str):
            try:
                _cost_function_class, _kwargs = STRING_TO_COST_FUNCTION[cost_function]
            except KeyError:
                raise XYFitException('Unknown cost function: %s' % cost_function)
            self._cost_function = _cost_function_class(**_kwargs)
        else:
            self._cost_function = CostFunction(cost_function)
            # self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?

        # validate x error algorithm
        if x_error_algorithm not in XYFit.X_ERROR_ALGORITHMS:
            raise ValueError("Unknown value for 'x_error_algorithm': "
                             "{}. Expected one of:".format(x_error_algorithm,
                                                           ', '.join(['iterative linear', 'nonlinear'])))
        self._x_error_algorithm = x_error_algorithm

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
        self.data = xy_data


    # -- private methods

    def _init_nexus(self):
        self._nexus = Nexus()

        for _axis in ('x', 'y'):
            for _type in ('data', 'model'):

                # add data and model for axis
                self._add_property_to_nexus('_'.join((_axis, _type)))
                # add errors for axis
                self._add_property_to_nexus('_'.join((_axis, _type, 'error')))

                # add cov mats for axis
                for _prop in ('cov_mat', 'uncor_cov_mat'):
                    _node = self._add_property_to_nexus('_'.join((_axis, _type, _prop)))
                    # add inverse
                    self._nexus.add_function(
                        invert_matrix,
                        func_name='_'.join((_node.name, 'inverse')),
                        par_names=(_node.name,)
                    )

            # 'total_error', i.e. data + model error in quadrature
            self._nexus.add_function(
                add_in_quadrature,
                func_name='_'.join((_axis, 'total', 'error')),
                par_names=(
                    '_'.join((_axis, 'data', 'error')),
                    '_'.join((_axis, 'model', 'error'))
                )
            )

            # 'total_cov_mat', i.e. data + model cov mats
            for _mat in ('cov_mat', 'uncor_cov_mat'):
                _node = (
                    self._nexus.get('_'.join((_axis, 'data', _mat))) +
                    self._nexus.get('_'.join((_axis, 'model', _mat)))
                )
                _node.name = '_'.join((_axis, 'total', _mat))
                self._nexus.add(_node)

                # add inverse
                self._nexus.add_function(
                    invert_matrix,
                    func_name='_'.join((_node.name, 'inverse')),
                    par_names=(_node.name,),
                )

        # nuisance parameter-related properties
        self._add_property_to_nexus(
            '_y_data_nuisance_cor_design_mat',
            #obj=self._data_container,
            #name='_y_data_nuisance_cor_design_mat'
        )
        self._add_property_to_nexus(
            '_y_model_nuisance_cor_design_mat',
            #obj=self._param_model,
            #name='_y_model_nuisance_cor_design_mat'
        )

        self._nexus.add_alias(
            '_y_total_nuisance_cor_design_mat',
            alias_for='_y_data_nuisance_cor_design_mat'
        )

        _node = self._add_property_to_nexus('projected_xy_total_error')
        _node = self._add_property_to_nexus('projected_xy_total_cov_mat')
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

            self._fit_param_names.append(_par_name)
            self._poi_names.append(_par_name)

        self._poi_names = tuple(self._poi_names)

        # -- nuisance parameters
        self._y_nuisance_names = []  # names of all nuisance parameters accounting for correlated y errors
        self._x_uncor_nuisance_names = []  # names of all nuisance parameters accounting for uncorrelated x errors
        # TODO
        # self._x_cor_nuisance_names = []  # names of all nuisance parameters accounting for correlated x errors

        self._nexus.add_function(lambda: self.poi_values, func_name='poi_values')
        self._nexus.add_function(lambda: self.parameter_values, func_name='parameter_values')
        self._nexus.add_function(lambda: self.parameter_constraints, func_name='parameter_constraints')

        # add the original function name as an alias to 'y_model'
        try:
            self._nexus.add_alias(self._model_function.name, alias_for='y_model')
        except NexusError:
            pass  # allow 'y_model' as function name for model function

        self._nexus.add_function(
            collect,
            func_name="y_nuisance_vector"
        )
        self._nexus.add_function(
            collect,
            func_name="x_uncor_nuisance_vector"
        )

        # -- initialize nuisance parameters

        # TODO: reimplement nuisance parameters

        ### # one nuisance parameter per correlated 'y' error
        ### if self._cost_function.get_flag("need_y_nuisance") and self._data_container.has_y_errors:
        ###     # retrieve the errors for which to assign 'y'-nuisance parameters
        ###     _nuisance_error_objects = self.get_matching_errors(
        ###         matching_criteria=dict(
        ###             axis=1,  # cannot use 'y' here
        ###             correlated=True
        ###         )
        ###     )
        ###     for _err_name, _err_obj in six.iteritems(_nuisance_error_objects):
        ###         _nuisance_name = "_n_yc_{}".format(_err_name)
        ###         self._nexus.new(**{_nuisance_name: 0.0})
        ###         self._nexus.add_dependency(_nuisance_name, "y_nuisance_vector")
        ###         self._fit_param_names.append(_nuisance_name)
        ###         self._y_nuisance_names.append(_nuisance_name)
        ###     self._nexus.set_function_parameter_names("y_nuisance_vector", self._y_nuisance_names)
        ###
        ### # one 'x' nuisance parameter per data point (TODO: and one per correlated 'x' error)
        ### if self._cost_function.get_flag("need_x_nuisance") and self._data_container.has_uncor_x_errors:
        ###     # one 'x' nuisance parameter per data point
        ###     for i in six.moves.range(self._data_container.size):
        ###         _nuisance_name = "_n_xu_{}".format(i)
        ###         self._nexus.new(**{_nuisance_name: 0.0})
        ###         self._nexus.add_dependency(_nuisance_name, "x_uncor_nuisance_vector")
        ###         self._fit_param_names.append(_nuisance_name)
        ###         self._x_uncor_nuisance_names.append(_nuisance_name)
        ###     self._nexus.set_function_parameter_names("x_uncor_nuisance_vector", self._x_uncor_nuisance_names)
        ###     # TODO
        ###     # # retrieve the errors for which to assign 'x'-nuisance parameters
        ###     # _nuisance_error_objects = self.get_matching_errors(
        ###     #     matching_criteria=dict(
        ###     #         axis=0,  # cannot use 'x' here
        ###     #         correlated=True
        ###     #     )
        ###     # )
        ###     # for _err_name, _err_obj in six.iteritems(_nuisance_error_objects):
        ###     #     _nuisance_name = "_n_xc_{}".format(_err_name)
        ###     #     self._nexus.new(**{_nuisance_name: 0.0})
        ###     #     self._nexus.add_dependency(_nuisance_name, "x_cor_nuisance_vector")
        ###     #     self._fit_param_names.append(_nuisance_name)
        ###     #     self._x_cor_nuisance_names.append(_nuisance_name)
        ###     # self._nexus.set_function_parameter_names("x_cor_nuisance_vector", self._x_cor_nuisance_names)

        # the cost function (the function to be minimized)
        _cost_node = self._nexus.add_function(
            self._cost_function,
            par_names=self._cost_function.arg_names,
            func_name=self._cost_function.name,
        )

        _cost_alias = self._nexus.add_alias('cost', alias_for=self._cost_function.name)

        self._nexus.add_dependency('poi_values', depends_on=self._poi_names)
        self._nexus.add_dependency('parameter_values', depends_on=self._fit_param_names)

        self._nexus.add_dependency(
            'projected_xy_total_cov_mat',
            depends_on=(
                'poi_values',
                'x_model',
                'x_total_cov_mat',
                'y_total_cov_mat'
            )
        )
        self._nexus.add_dependency(
            'projected_xy_total_error',
            depends_on=(
                'poi_values',
                'x_model',
                'x_total_error',
                'y_total_error'
            )
        )

        self._nexus.add_dependency(
            'y_model',
            depends_on=(
                'x_model',
                'poi_values'
            )
        )

        self._nexus.add_dependency(
            'x_model',
            depends_on=(
                'x_data',
            )
        )

        for _axis in ('x', 'y'):
            for _side in ('data', 'model', 'total'):
                self._nexus.add_dependency(
                    '{}_{}_uncor_cov_mat'.format(_axis, _side),
                    depends_on='{}_{}_cov_mat'.format(_axis, _side)
                )

        # in case 'x' errors are defined and the corresponding
        # algorithm is 'iterative linear', matrices should be projected
        # once and the corresponding node made frozen
        if (self._x_error_algorithm == 'iterative linear' and
            not self.has_x_errors):

            self._with_projected_nodes(('update', 'freeze'))

    def _with_projected_nodes(self, actions):
        '''perform actions on projected error nodes: freeze, update, unfreeze...'''
        if isinstance(actions, str):
            actions = (actions,)
        for _node_name in ('projected_xy_total_error', 'projected_xy_total_cov_mat'):
            for action in actions:
                _node = self._nexus.get(_node_name)
                getattr(_node, action)()

    def _get_poi_index_by_name(self, name):
        try:
            return self._poi_names.index(name)
        except ValueError:
            raise self.EXCEPTION_TYPE('Unknown parameter name: %s' % name)

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise XYFitException("Incompatible container type '%s' (expected '%s')"
                                 % (type(new_data), self.CONTAINER_TYPE))
        else:
            _x_data = new_data[0]
            _y_data = new_data[1]
            self._data_container = self._new_data_container(_x_data, _y_data, dtype=float)
        self._data_container._on_error_change_callback = self._on_error_change

        # update nexus data nodes
        self._nexus.get('x_data').mark_for_update()
        self._nexus.get('y_data').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = self._new_parametric_model(
            self.x_model,
            self._model_function,
            self.poi_values
        )
        self._param_model._on_error_change_callbacks = [self._on_error_change]

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
        # call base method to show header and model function
        super(XYFit, self)._report_model(output_stream, indent, indentation_level)
        # print model values at POIs
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
        # if cost function uses x-nuisance parameters, consider these
        #if self._cost_function.get_flag("need_x_nuisance") and self._data_container.has_uncor_x_errors:
        #    return self.x_data + (self.x_uncor_nuisance_values * self.x_data_error)
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
        return add_in_quadrature(self.x_model_error, self.x_data_error)

    @property
    def y_total_error(self):
        """array of pointwise total *y* uncertainties"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return add_in_quadrature(self.y_model_error, self.y_data_error)

    @property
    def projected_xy_total_error(self):
        """array of pointwise total *y* with the x uncertainties projected on top of them"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_error

        _x_errors = self.x_total_error
        _precision = 0.01 * np.min(_x_errors)
        _derivatives = self._param_model.eval_model_function_derivative_by_x(
            dx=_precision,
            model_parameters=self.parameter_values
        )

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
        _derivatives = self._param_model.eval_model_function_derivative_by_x(
            dx=_precision,
            model_parameters=self.parameter_values
        )
        _outer_product = np.outer(_derivatives, _derivatives)

        return self.y_total_cov_mat + self.x_total_cov_mat * _outer_product


    @property
    def x_total_cov_mat_inverse(self):
        """inverse of the total *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.x_total_cov_mat)
        except np.linalg.LinAlgError:
            return None

    @property
    def y_total_cov_mat_inverse(self):
        """inverse of the total *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.y_total_cov_mat)
        except np.linalg.LinAlgError:
            return None

    @property
    def projected_xy_total_cov_mat_inverse(self):
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.projected_xy_total_cov_mat)
        except np.linalg.LinAlgError:
            return None

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

        try:
            return invert_matrix(self.y_total_uncor_cov_mat)
        except np.linalg.LinAlgError:
            return None

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

        return self.x_data_uncor_cov_mat + self.x_model_uncor_cov_mat

    @property
    def x_total_uncor_cov_mat_inverse(self):
        """inverse of the total *x* uncorrelated covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model

        try:
            return invert_matrix(self.x_total_uncor_cov_mat)
        except np.linalg.LinAlgError:
            return None

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
        for _name in self.poi_names:
            _poi_values.append(self.parameter_name_value_dict[_name])
        return np.array(_poi_values)

    @property
    def poi_names(self):
        return self._poi_names

    @property
    def x_uncor_nuisance_values(self):
        """gives the x uncorrelated nuisance vector"""
        _values = []
        for _name in self._x_uncor_nuisance_names:
            _values.append(self.parameter_name_value_dict[_name])
        return np.asarray(_values)

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

    def set_poi_values(self, param_values):
        """set the start values of all parameters of interests"""
        _param_names = self._poi_names
        #test list length
        if not len(param_values) == len(_param_names):
            raise XYFitException("Cannot set all fit parameter values: %d fit parameters declared, "
                                       "but %d provided!"
                                       % (len(_param_names), len(param_values)))
        # set values in nexus
        _par_val_dict = {_pn: _pv for _pn, _pv in zip(_param_names, param_values)}
        self.set_parameter_values(**_par_val_dict)

    def do_fit(self, asymmetric_parameter_errors=False):
        if self._cost_function.needs_errors and not self._data_container.has_y_errors:
            self._cost_function.on_no_errors()

        # explicitly update (frozen) projected covariance matrix before fit
        self._with_projected_nodes('update')

        if self.has_x_errors:
            if self._x_error_algorithm == 'nonlinear':
                # 'nonlinear' x error fitting: one iteration;
                # projected covariance matrix is updated during minimization
                self._with_projected_nodes(('update', 'unfreeze'))
                super(XYFit, self).do_fit()

            elif self._x_error_algorithm == 'iterative linear':
                # 'iterative linear' x error fitting: multiple iterations;
                # projected covariance matrix is only updated in-between
                # and kept constant during minimization
                self._with_projected_nodes(('update', 'freeze'))

                # perform a preliminary fit
                self._fitter.do_fit()

                # iterate until cost function value converges
                _convergence_limit = float(kc('fit', 'x_error_fit_convergence_limit'))
                _previous_cost_function_value = self.cost_function_value
                for i in range(kc('fit', 'max_x_error_fit_iterations')):

                    # explicitly update (frozen) projected matrix before each iteration
                    self._with_projected_nodes('update')

                    self._fitter.do_fit()

                    # check convergence
                    if np.abs(self.cost_function_value - _previous_cost_function_value) < _convergence_limit:
                        break  # fit converged

                    _previous_cost_function_value = self.cost_function_value

        else:
            # no 'x' errors: fit as usual

            # freeze error projection nodes (faster)
            self._with_projected_nodes(('update', 'freeze'))

            super(XYFit, self).do_fit()

        # explicitly update error projection nodes
        self._with_projected_nodes('update')

        # unfreeze error projection nodes only if fit has x errors
        if self.has_x_errors:
            self._with_projected_nodes('unfreeze')

        # clear loaded results and update parameter formatters
        self._loaded_result_dict = None
        self._update_parameter_formatters()
        return self.get_result_dict(asymmetric_parameter_errors=asymmetric_parameter_errors)

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
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.x_model
        return self._param_model.eval_model_function_derivative_by_parameters(x=x, model_parameters=model_parameters)

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
