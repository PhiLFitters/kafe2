import abc
import sys
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import six

from .container import DataContainerBase, DataContainerException
from ..io.file import FileIOMixin
from ...config import kc
from ...core.fitters.nexus import Nexus, NexusError, Parameter, Array
from ...core.fitters.nexus_fitter import NexusFitter
from ...core.constraint import GaussianMatrixParameterConstraint, GaussianSimpleParameterConstraint
from ...core.error import CovMat
from ...tools import print_dict_as_table
from .._base.cost import CostFunction, STRING_TO_COST_FUNCTION
from ..util import invert_matrix, add_in_quadrature, cholesky_decomposition, log_determinant

__all__ = ["FitBase", "FitException"]


class FitException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class FitBase(FileIOMixin, object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    types of fitters.
    """

    CONTAINER_TYPE = DataContainerBase
    MODEL_TYPE = None
    MODEL_FUNCTION_TYPE = None
    PLOT_ADAPTER_TYPE = None
    EXCEPTION_TYPE = FitException
    RESERVED_NODE_NAMES = None
    _BASIC_ERROR_NAMES = {}
    _STRING_TO_COST_FUNCTION = STRING_TO_COST_FUNCTION
    _AXES = (None,)  # axes for which to for example create data nexus nodes
    _MODEL_NAME = "model"
    _MODEL_ERROR_NODE_NAMES = ["model_error", "model_cov_mat"]

    def __init__(
            self, data, model_function, cost_function, minimizer=None, minimizer_kwargs=None,
            dynamic_error_algorithm="nonlinear"):
        """This is a purely abstract class implementing the minimal interface required by all
        types of fits.

        :param minimizer: Name of the minimizer to use.
        :type minimizer: str or None
        :param minimizer_kwargs: Dictionary wit keywords for initializing the minimizer.
        :type minimizer_kwargs: dict or None
        :param dynamic_error_algorithm: how to handle errors that depend on the model parameters.
        :type dynamic_error_algorithm: "nonlinear" or "iterative".
        """
        super(FitBase, self).__init__()
        self._data_container = None
        self._param_model = None
        self._nexus = None
        self._fitter = None
        self._fit_param_names = []  # names of all fit parameters
        self._fit_param_constraints = []
        self._loaded_result_dict = None  # contains potential fit results from a file or multifit

        # save minimizer, minimizer_kwargs for serialization
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs

        self.dynamic_error_algorithm = dynamic_error_algorithm
        self._dynamic_error_warning_printed = False

        # set/construct the model function object
        if self.MODEL_FUNCTION_TYPE is None:
            if model_function is not None:
                raise ValueError(
                    "Model function must be None if subclass does not define MODEL_FUNCTION_TYPE.")
            self._model_function = model_function
        else:
            if isinstance(model_function, self.MODEL_FUNCTION_TYPE):
                self._model_function = model_function
            else:
                self._model_function = self.MODEL_FUNCTION_TYPE(model_function)

            # Disallow using reserved keywords as model function arguments:
            if not self.RESERVED_NODE_NAMES.isdisjoint(
                    set(self._model_function.signature.parameters)):
                _invalid_args = self.RESERVED_NODE_NAMES.intersection(
                    set(self._model_function.signature.parameters))
                raise self.__class__.EXCEPTION_TYPE(
                    "The following names are reserved and cannot be used as model function arguments: %r"
                    % (_invalid_args,))

        # set and validate the cost function
        if isinstance(cost_function, CostFunction):
            self._cost_function = cost_function
        elif isinstance(cost_function, str):
            try:
                _cost_function_class, _kwargs = self._STRING_TO_COST_FUNCTION[cost_function]
            except KeyError:
                raise ValueError('Unknown cost function: %s' % cost_function)
            self._cost_function = _cost_function_class(**_kwargs)
        elif cost_function is not None:
            self._cost_function = CostFunction(cost_function)
            # self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?
        else:
            self._cost_function = None

        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._initialize_fitter()

        # set the data after the cost_function has been set and nexus has been initialized
        if data is not None:
            self.data = data

    # -- private methods

    def _add_property_to_nexus(self, prop, obj=None, name=None, depends_on=None):
        """register a property of this object in the nexus as a function node"""
        obj = obj if obj is not None else self
        _node = self._nexus.add_function(
            partial(getattr(obj.__class__, prop).fget, obj),
            func_name=name or prop
        )
        if depends_on is not None:
            self._nexus.add_dependency(name=_node.name, depends_on=depends_on)
        return _node

    @classmethod
    def _get_base_class(cls):
        return FitBase

    @classmethod
    def _get_object_type_name(cls):
        return 'fit'

    def _init_nexus(self):
        self._nexus = Nexus()

        # -- fit parameters

        _parameter_nodes = []
        # get names and default values of all parameters if the fit has a model function
        if self._model_function is not None:
            for _par_name, _par_value in six.iteritems(self._model_function.defaults_dict):
                # create nexus node for function parameter
                _parameter_nodes.append(self._nexus.add(Parameter(_par_value, name=_par_name)))

                self._fit_param_names.append(_par_name)

        self._nexus.add(Array(_parameter_nodes, name="parameter_values"))
        self._nexus.add_function(lambda: self.parameter_constraints,
                                 func_name='parameter_constraints')

        # -- errors

        for _axis in self._AXES:
            _error_names = []
            _mat_names = []
            for _type in ("model", "data", "total"):
                _name = '_'.join((_axis, _type)) if _axis is not None else _type
                _error_name = '_'.join((_axis, _type, 'error')) if _axis is not None \
                    else '_'.join((_type, 'error'))
                _mat_name = '_'.join((_axis, _type, "cov_mat")) if _axis is not None \
                    else '_'.join((_type, "cov_mat"))
                if _type == "total":
                    # Performance optimization: calculate total foo in Nexus
                    self._nexus.add_function(
                        lambda m, d: np.sqrt(m ** 2 + d ** 2),
                        func_name=_error_name, par_names=_error_names)
                    self._nexus.add_function(
                        lambda m, d: m + d, func_name=_mat_name, par_names=_mat_names)
                else:
                    self._add_property_to_nexus(_name)
                    self._add_property_to_nexus(_error_name)
                    _error_names.append(_error_name)
                    self._add_property_to_nexus(_mat_name)
                    _mat_names.append(_mat_name)
                if _error_name in self._MODEL_ERROR_NODE_NAMES:
                    self._nexus.add_dependency(_error_name, depends_on="parameter_values")
                if _mat_name in self._MODEL_ERROR_NODE_NAMES:
                    self._nexus.add_dependency(_mat_name, depends_on="parameter_values")
                self._add_property_to_nexus(_mat_name + "_inverse", depends_on=_mat_name)
                self._nexus.add_function(cholesky_decomposition, func_name=_mat_name + "_cholesky",
                                         par_names=[_mat_name])
                self._nexus.add_function(log_determinant, func_name=_mat_name + "_log_determinant",
                                         par_names=[_mat_name + "_cholesky"])

        if self._model_function is not None:
            # add the original function name as an alias to 'model'
            try:
                self._nexus.add_alias(self._model_function.name, alias_for=self._MODEL_NAME)
            except NexusError:
                pass  # allow 'model' as function name for model

        if self._cost_function is not None:
            # the cost function (the function to be minimized)
            _cost_node = self._nexus.add_function(
                self._cost_function,
                par_names=self._cost_function.arg_names,
                func_name=self._cost_function.name,
            )

            _cost_alias = self._nexus.add_alias('cost', alias_for=self._cost_function.name)

    def _initialize_fitter(self):
        self._fitter = NexusFitter(nexus=self._nexus,
                                   parameters_to_fit=self._fit_param_names,
                                   parameter_to_minimize=self._cost_function.name,
                                   minimizer=self._minimizer,
                                   minimizer_kwargs=self._minimizer_kwargs)

    @abc.abstractmethod
    def _set_new_data(self, new_data):
        """Private method called by the :py:attr:`~data` setter. Must be overwritten for each derived class."""
        pass

    @abc.abstractmethod
    def _set_new_parametric_model(self):
        pass

    # Overridden by MultiFit
    def _get_model_function_argument_formatters(self):
        """All arguments of the model function including independent variables."""
        return self._model_function.formatter.arg_formatters

    # Overridden by MultiFit
    def _get_model_function_parameter_formatters(self):
        """Only the parameters which are fitted. Excludes independent variables."""
        return self.model_function.formatter.par_formatters

    def _report_data(self, output_stream, indent, indentation_level):
        """Report the data used in this fit to the given output stream.

        :param io.TextIOBase output_stream: The output stream to which the report should be printed.
        :param str indent: The string to use when indenting lines.
        :param int indentation_level: How many times the indent str should be placed before each line.
        """
        pass

    def _report_model(self, output_stream, indent, indentation_level):
        """Report the fit model to the given output stream.

        :param io.TextIOBase output_stream: The output stream to which the report should be printed.
        :param str indent: The string to use when indenting lines.
        :param int indentation_level: How many times the indent str should be placed before each line.
        """
        output_stream.write(indent * indentation_level + '#########\n')
        output_stream.write(indent * indentation_level + '# Model #\n')
        output_stream.write(indent * indentation_level + '#########\n\n')
        output_stream.write(indent * (indentation_level + 1) + "Model Function\n")
        output_stream.write(indent * (indentation_level + 1) + "==============\n\n")
        output_stream.write(indent * (indentation_level + 2))
        output_stream.write(self._model_function.formatter.get_formatted(with_expression=True))
        output_stream.write('\n\n')

    def _report_fit_results(self, output_stream, indent, indentation_level, asymmetric_parameter_errors):
        """Report the fit results

        :param io.TextIOBase output_stream: The output stream to which the report should be printed.
        :param str indent: The string to use when indenting lines.
        :param int indentation_level: How many times the indent str should be placed before each line.
        :param bool asymmetric_parameter_errors: If asymmetric parameter uncertainties should be used instead of
            symmetric uncertainties.
        """
        output_stream.write(indent * indentation_level + '###############\n')
        output_stream.write(indent * indentation_level + '# Fit Results #\n')
        output_stream.write(indent * indentation_level + '###############\n\n')

        if not self.did_fit:
            output_stream.write(indent * (indentation_level + 1) +
                                'WARNING: No fit has been performed as of yet. Did you forget to run fit.do_fit()?\n\n')

        output_stream.write(indent * (indentation_level + 1) + "Model Parameters\n")
        output_stream.write(indent * (indentation_level + 1) + "================\n\n")

        self._update_parameter_formatters(update_asymmetric_errors=asymmetric_parameter_errors)
        for _pf in self._get_model_function_parameter_formatters():
            output_stream.write(indent * (indentation_level + 2))
            output_stream.write(
                _pf.get_formatted(with_name=True,
                                  with_value=True,
                                  with_errors=True,
                                  format_as_latex=False,
                                  asymmetric_error=asymmetric_parameter_errors)
            )
            output_stream.write('\n')
        output_stream.write('\n')

        output_stream.write(indent * (indentation_level + 1) + "Model Parameter Correlations\n")
        output_stream.write(indent * (indentation_level + 1) + "============================\n\n")

        _cor_mat_content = self.parameter_cor_mat
        if _cor_mat_content is not None:
            par_display_names = [_pf.name for _pf in
                                 self._get_model_function_parameter_formatters()]
            _cor_mat_as_dict = OrderedDict()
            _cor_mat_as_dict['_invisible_first_column'] = par_display_names
            for _par_name, _row in zip(par_display_names, self.parameter_cor_mat.T):
                _cor_mat_as_dict[_par_name] = np.atleast_1d(np.squeeze(np.asarray(_row)))

            print_dict_as_table(_cor_mat_as_dict, output_stream=output_stream, indent_level=2)
        else:
            output_stream.write(indent * (indentation_level + 2) + '<not available>\n')
        output_stream.write('\n')

        output_stream.write(indent * (indentation_level + 1) + "Cost Function\n")
        output_stream.write(indent * (indentation_level + 1) + "=============\n\n")

        _pf = self._cost_function.formatter
        output_stream.write(
            indent * (indentation_level + 2) + "Cost function: {}\n\n".format(_pf.description))
        _gof_value = self.goodness_of_fit
        output_stream.write(indent * (indentation_level + 2))
        if _gof_value is None:
            output_stream.write("Cost = ")
            output_stream.write(_pf.get_formatted(
                value=self.cost_function_value,
                with_name=False,
                format_as_latex=False
            ))
        else:
            _gof_string = "chi2 / ndf = " if self._cost_function.is_chi2 else "GoF / ndf = "
            output_stream.write(_gof_string)
            output_stream.write(_pf.get_formatted(
                value=_gof_value,
                n_degrees_of_freedom=self.ndf,
                with_name=False,
                with_value_per_ndf=True,
                format_as_latex=False
            ))
        output_stream.write('\n\n')
        _chi2_prob = self.chi2_probability
        if _chi2_prob is not None:
            output_stream.write("%schi2 probability = %#.3g\n\n" % (
                indent * (indentation_level + 2), _chi2_prob))

    def _update_parameter_formatters(self, update_asymmetric_errors=False):
        """Update all parameter formatters with the current values and uncertainties.

        :param bool update_asymmetric_errors: If the asymmetric parameter uncertainties should be updated as well.
        """
        for _fpf, _pv, _pe in zip(
                self._get_model_function_parameter_formatters(), self.parameter_values, self.parameter_errors):
            _fpf.value = _pv
            _fpf.error = _pe
        if update_asymmetric_errors:
            self._check_dynamic_error_compatibility()
            for _fpf, _ape in zip(self._get_model_function_parameter_formatters(), self.asymmetric_parameter_errors):
                _fpf.asymmetric_error = _ape

    def _on_error_change(self):
        """Mark all error nodes in :py:attr:`~_BASIC_ERROR_NAMES` for updates in the nexus."""
        self._fitter.reset_minimizer()
        for _error_name in self._BASIC_ERROR_NAMES:
            self._nexus.get(_error_name).mark_for_update()

    def _set_data_as_model_ref(self):
        for _err in self._param_model.get_matching_errors({"relative": True}).values():
            _old_ref = _err.reference
            _err.reference = self._data_container.data

    def _iterative_fits_needed(self):
        return bool(self._param_model.get_matching_errors({"relative": True})) \
               and self._dynamic_error_algorithm == "iterative"

    def _second_fit_needed(self):
        return bool(self._param_model.get_matching_errors({"relative": True})) \
               and self._dynamic_error_algorithm == "nonlinear"

    def _get_node_names_to_freeze(self, first_fit):
        if first_fit or not self._param_model.get_matching_errors({"relative": True}) \
                or self._dynamic_error_algorithm == "iterative":
            return self._MODEL_ERROR_NODE_NAMES
        else:
            return []

    def _pre_fit_iteration(self, first_fit=False):
        for _model_err_name in self._get_node_names_to_freeze(first_fit):
            _node = self._nexus.get(_model_err_name)
            _node.update()
            _node.freeze()

    def _post_fit_iteration(self, first_fit=False):
        for _model_err_name in self._get_node_names_to_freeze(first_fit):
            _node = self._nexus.get(_model_err_name)
            _node.unfreeze()
            _node.update()
            _node.notify_parents()

    def _check_dynamic_error_compatibility(self):
        if not self._dynamic_error_warning_printed and self._iterative_fits_needed():
            warnings.warn(
                "Asymmetric parameter errors, parameter profiles, and contours cannot be "
                "calculated with iterative dynamic errors. Used nonlinear errors instead.")
            self._dynamic_error_warning_printed = True

    # -- public properties

    @property
    def data(self):
        """array of measurement values"""
        return self._data_container.data

    @data.setter
    def data(self, new_data):
        self._set_new_data(new_data)
        # validate cost function
        _data_and_cost_compatible, _reason = self._cost_function.is_data_compatible(self.data)
        if not _data_and_cost_compatible:
            raise self.EXCEPTION_TYPE('Fit data and cost function are not compatible: %s' % _reason)
        self._set_new_parametric_model()
        self._param_model._on_error_change_callbacks = [self._on_error_change]

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
    def data_container(self):
        """The data container used in this fit.

        :rtype: kafe2.fit._base.DataContainerBase
        """
        return self._data_container

    @property
    @abc.abstractmethod
    def model(self):
        pass

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
        return self._nexus.get("total_error").value

    @property
    def total_cov_mat(self):
        """the total covariance matrix"""
        return self._nexus.get("total_cov_mat").value

    @property
    def total_cov_mat_inverse(self):
        """inverse of the total covariance matrix (or ``None`` if singular)"""
        return invert_matrix(self.total_cov_mat)

    @property
    def total_cor_mat(self):
        """the total correlation matrix"""
        return CovMat(self.total_cov_mat).cor_mat

    @property
    def model_function(self):
        """The wrapped model function as a :py:obj:`~kafe2.fit._base.ModelFunctionBase` or derived object.
        This object contains the model function as well as formatting information used for this fit.

        :rtype: kafe2.fit._base.ModelFunctionBase"""
        return self._model_function

    @property
    def model_label(self):
        """The label of the model used in this fit.

        :rtype: str or None
        """
        return self._param_model.label

    @model_label.setter
    def model_label(self, label):
        self._param_model.label = label

    @property
    def parameter_values(self):
        """The current parameter values.

        :rtype: numpy.ndarray[float]
        """
        return self._nexus.get("parameter_values").value

    @property
    def parameter_names(self):
        """The current parameter names.

        :rtype: tuple[str]
        """
        return self._fitter.parameters_to_fit

    @property
    def parameter_errors(self):
        """The current parameter uncertainties.

        :rtype: numpy.ndarray[float]
        """
        if self._loaded_result_dict is not None:
            return self._loaded_result_dict['parameter_errors']
        return self._fitter.fit_parameter_errors

    @property
    def parameter_cov_mat(self):
        """The current parameter covariance matrix.

        :rtype: None or numpy.ndarray[numpy.ndarray[float]]
        """
        if self._loaded_result_dict is not None:
            return self._loaded_result_dict['parameter_cov_mat']
        return self._fitter.fit_parameter_cov_mat

    @property
    def parameter_cor_mat(self):
        """The current parameter correlation matrix.

        :rtype: None or numpy.ndarray[numpy.ndarray[float]]
        """
        if self._loaded_result_dict is not None:
            return self._loaded_result_dict['parameter_cor_mat']
        return self._fitter.fit_parameter_cor_mat

    @property
    def asymmetric_parameter_errors(self):
        """The current asymmetric parameter uncertainties.

        :rtype: numpy.ndarray[numpy.ndarray[float, float]]
        """
        if self._loaded_result_dict is not None and self._loaded_result_dict['asymmetric_parameter_errors'] is not None:
            return self._loaded_result_dict['asymmetric_parameter_errors']
        self._check_dynamic_error_compatibility()
        return self._fitter.asymmetric_fit_parameter_errors

    @property
    def parameter_name_value_dict(self):
        """A dictionary mapping each parameter name to its current value.

        :rtype: OrderedDict[str, float]
        """
        return self._fitter.get_fit_parameter_values()

    @property
    def parameter_constraints(self):
        """The gaussian constraints given for the fit parameters.

        :rtype: list[kafe2.core.constraint.GaussianSimpleParameterConstraint or
            kafe2.core.constraint.GaussianMatrixParameterConstraint]
        """
        return self._fit_param_constraints

    @property
    def cost_function_value(self):
        """The current value of the cost function.

        :rtype: float
        """
        return self._fitter.parameter_to_minimize_value

    @property
    def data_size(self):
        """The size (number of points) of the data container.

        :rtype: int
        """
        return self._data_container.size

    @property
    def has_model_errors(self):
        """:py:obj:`True` if at least one uncertainty source is defined for the model.

        :rtype: bool
        """
        return self._param_model.has_errors

    @property
    def has_data_errors(self):
        """:py:obj:`True` if at least one uncertainty source is defined for the data.

        :rtype: bool
        """
        return self._data_container.has_errors

    @property
    def has_errors(self):
        """:py:obj:`True` if at least one uncertainty source is defined for either the data or the model.

        :rtype: bool
        """
        return True if self.has_data_errors or self.has_model_errors else False

    @property
    def model_count(self):
        """The number of model functions contained in the fit, 1 by default.

        :rtype: int
        """
        return 1

    @property
    def did_fit(self):
        """Whether a fit was performed for the given data and model.

        :rtype: bool
        """
        if self._loaded_result_dict is not None:
            return self._loaded_result_dict['did_fit']
        return self._fitter.state_is_from_minimizer

    @property
    def ndf(self):
        """The degrees of freedom of this fit.

        :rtype: int
        """
        _extra_ndf_constraints = 0
        for _parameter_constraint in self._fit_param_constraints:
            _extra_ndf_constraints += _parameter_constraint.extra_ndf
        return self._param_model.ndf + len(self._fitter.fixed_parameters) + _extra_ndf_constraints

    @property
    def goodness_of_fit(self):
        return self._cost_function.goodness_of_fit(
            *[self._nexus.get(_node_name).value for _node_name in self._cost_function.arg_names])

    @property
    def dynamic_error_algorithm(self):
        """The algorithm to use for handling errors that depend on the model parameters.
        :rtype: str
        """
        return self._dynamic_error_algorithm

    @dynamic_error_algorithm.setter
    def dynamic_error_algorithm(self, new_dea):
        _valid_deas = ["nonlinear", "iterative"]
        if new_dea not in _valid_deas:
            raise ValueError(
                "Unknown dynamic error algorithm: %s. Valid algorithms: %s" % (
                    new_dea, _valid_deas))
        self._dynamic_error_algorithm = new_dea

    @property
    def chi2_probability(self):
        """The chi2 probability for the current model values."""
        _cost = self.cost_function_value
        if self._cost_function.add_determinant_cost:
            _cost -= self._nexus.get("total_cov_mat_log_determinant").value
        return self._cost_function.chi2_probability(_cost, self.ndf)

    # -- public methods

    def set_parameter_values(self, **param_name_value_dict):
        """Set the fit parameters to new values. Valid keyword arguments are the names of the declared fit parameters.

        :param param_name_value_dict: new parameter values
        """
        _return_value = self._fitter.set_fit_parameter_values(**param_name_value_dict)
        if self._param_model is not None:
            self._param_model.parameters = self.parameter_values
        return _return_value

    def set_all_parameter_values(self, param_value_list):
        """Set all the fit parameters at the same time.

        :param typing.Iterable[float] param_value_list: List of parameter values (mind the order).
        """
        if self._param_model is not None:
            self._param_model.parameters = param_value_list
        return self._fitter.set_all_fit_parameter_values(param_value_list)

    def fix_parameter(self, name, value=None):
        """Fix a parameter so that its value doesn't change when calling :py:meth:`~do_fit()`.

        :param str name: The name of the parameter to be fixed
        :param float or None value: The value to be given to the fixed parameter. If :py:obj:`None` the current value
            from :py:attr:`~parameter_values` will be used.
        """
        self._fitter.fix_parameter(name=name, value=value)
        _par_index = self.parameter_names.index(name)
        self._get_model_function_parameter_formatters()[_par_index].fixed = True

    def release_parameter(self, name):
        """Release a fixed parameter so that its value once again changes when calling :py:meth:`~do_fit()`.

        :param str name: The name of the fixed parameter to be released
        """
        self._fitter.release_parameter(name=name)
        _par_index = self.parameter_names.index(name)
        self._get_model_function_parameter_formatters()[_par_index].fixed = False

    def limit_parameter(self, name, lower=None, upper=None):
        """Limit a parameter to a given range.

        :param str name: The name of the parameter to limit.
        :param float lower: The minimum parameter value.
        :param float upper: The maximum parameter value.
        """
        if lower is None and upper is None:
            raise ValueError("Either a lower or an upper bound must be provided!")

        # make sure these are numeric values (otherwise minimizer will
        # fail on `do_fit` with a cryptic error)
        for _lim in (lower, upper):
            try:
                assert _lim is None or float(_lim) == _lim
            except (TypeError, ValueError, AssertionError):
                six.raise_from(
                    TypeError("Expecting `None` or numeric value for parameter limit, got {}: {}".format(type(_lim), repr(_lim))),
                    None
                )

        self._fitter.limit_parameter(name=name, limits=(lower, upper))

    def unlimit_parameter(self, name):
        """Unlimit a parameter.

        :param str name: The name of the parameter to unlimit.
        """
        self._fitter.unlimit_parameter(name=name)

    def add_matrix_parameter_constraint(self, names, values, matrix, matrix_type='cov', uncertainties=None,
                                        relative=False):
        """Advanced class for applying correlated constraints to several parameters of a fit.
        The order of **names**, **values**, **matrix**, and **uncertainties** must be aligned.
        In other words the first index must belong to the first value, the first row/column in the matrix, etc.

        Let N be the number of parameters to be constrained.

        :param names: The names of the parameters to be constrained. Must be of shape (N,).
        :type names: typing.Collection[str]
        :param values: The values to which the parameters should be constrained. Must be of shape shape (N,).
        :type values: typing.Sized[float]
        :param matrix: The matrix that defines the correlation between the parameters. By default interpreted as a
            covariance matrix. Can also be interpreted as a correlation matrix by setting **matrix_type**.
            Must be of shape shape (N, N).
        :type matrix: typing.Iterable[float]
        :param matrix_type: Either ``'cov'`` or ``'cor'``. Defines whether the matrix should be interpreted as a
            covariance matrix or as a correlation matrix.
        :type matrix_type: str
        :param uncertainties: The uncertainties to be used in conjunction with a correlation matrix.
            Must be of shape (N,)
        :type uncertainties: None or typing.Iterable[float]
        :param relative: Whether the covariance matrix/the uncertainties should be interpreted as relative to
            **values**.
        :type relative: bool
        """
        if len(names) != len(values):
            raise self.EXCEPTION_TYPE(
                'Lengths of names and values are different: %s <-> %s' % (len(names), len(values)))
        _par_indices = []
        for _name in names:
            try:
                _par_indices.append(self.parameter_names.index(_name))
            except ValueError:
                raise self.EXCEPTION_TYPE('Unknown parameter name: %s' % _name)
        self._fit_param_constraints.append(GaussianMatrixParameterConstraint(
            indices=_par_indices, values=values, matrix=matrix, matrix_type=matrix_type, uncertainties=uncertainties,
            relative=relative
        ))

    def add_parameter_constraint(self, name, value, uncertainty, relative=False):
        """Apply a simple gaussian constraint to a single fit parameter.

        :param str name: The name of the parameter to be constrained.
        :param float value: The value to which the parameter should be constrained.
        :param float uncertainty: The uncertainty with which the parameter should be constrained to the given value.
        :param bool relative: Whether the given uncertainty is relative to the given value.
        """
        try:
            _index = self.parameter_names.index(name)
        except ValueError:
            raise self.EXCEPTION_TYPE('Unknown parameter name: %s' % name)
        self._fit_param_constraints.append(GaussianSimpleParameterConstraint(
            index=_index, value=value, uncertainty=uncertainty, relative=relative
        ))

    def get_matching_errors(self, matching_criteria=None, matching_type='equal'):
        """Return a list of uncertainty objects fulfilling the specified matching criteria.

        Valid keys for **matching_criteria**:
            * ``name`` (the unique error name)
            * ``type`` (either ``'simple'`` or ``'matrix'``)
            * ``correlated`` (bool, only matches simple errors!)
            * ``reference`` (either ``'model'`` or ``'data'``)

        .. note::
            The error objects contained in the dictionary are not copies, but the original error objects.
            Modifying them is possible, but not recommended.
            If you do modify any of them, the changes will not be reflected in the total error calculation until the
            error cache is cleared. This can be done by calling the private dataset method
            :py:meth:`~kafe2.fit._base.DataContainerBase._clear_total_error_cache`.

        :param matching_criteria: Key-value pairs specifying matching criteria. The resulting error array will only
                                  contain error objects matching *all* provided criteria.
                                  If :py:obj:`None`, all error objects are returned.
        :type matching_criteria: dict or None
        :param matching_type: How to perform the matching.
                              If ``'equal'``, the value in **matching_criteria** is checked for equality against the
                              stored value.
                              If ``'regex'``, the value in **matching_criteria** is interpreted as a regular expression
                              and is matched against the stored value.
        :type matching_type: str
        :return: Dict mapping error name to :py:obj:`~kafe2.core.error.GaussianErrorBase`-derived error objects.
        :rtype: dict[str, kafe2.core.error.GaussianErrorBase]
        """
        if matching_criteria is not None:
            _crit_ref_value = matching_criteria.pop('reference', None)
            if _crit_ref_value == 'data':
                return self._data_container.get_matching_errors(matching_criteria, matching_type=matching_type)
            if _crit_ref_value == 'model':
                return self._param_model.get_matching_errors(matching_criteria, matching_type=matching_type)
            if _crit_ref_value is None:
                pass  # don't raise, continue evaluation below
            else:
                raise ValueError("Unknown value '{}' for matching "
                                 "criterion 'reference'. Valid: 'data', 'model' or None".format(_crit_ref_value))

        _result = self._data_container.get_matching_errors(matching_criteria, matching_type=matching_type)
        _result_model = self._param_model.get_matching_errors(matching_criteria, matching_type=matching_type)

        # be paranoid about collisions
        for _k in _result_model:
            assert _k not in _result  # FATAL: there is an error with the same name in the data and model containers
            _result[_k] = _result_model[_k]

        return _result

    def add_error(self, err_val, name=None, correlation=0, relative=False, reference='data', **kwargs):
        """Add an uncertainty source to the fit.

        :param err_val: Pointwise uncertainty/uncertainties for all data points.
        :type err_val: float or typing.Iterable[float]
        :param name: Unique name for this uncertainty source. If :py:obj:`None`, the name of the error source will be
                     set to a random alphanumeric string.
        :type name: str or None
        :param correlation: Correlation coefficient between any two distinct data points.
        :type correlation: float
        :param relative: If :py:obj:`True`, **err_val** will be interpreted as a *relative* uncertainty.
        :type relative: bool
        :param reference: Either ``'data'`` or ``'model'``. Specifies which reference values to use when calculating
                          absolute errors from relative errors.
        :type reference: str
        :return: An error id which uniquely identifies the created error source.
        :rtype: str
        """
        if reference == 'data':
            # delegate to data container
            _reference_object = self._data_container
        elif reference == 'model':
            # delegate to model container
            _reference_object = self._param_model
        else:
            raise FitException("Cannot add error: unknown reference "
                               "specification '{}', expected one of: 'data', 'model'...".format(reference))

        _ret = _reference_object.add_error(err_val=err_val,
                                           name=name, correlation=correlation, relative=relative, **kwargs)

        return _ret

    def add_matrix_error(self, err_matrix, matrix_type,
                         name=None, err_val=None, relative=False, reference='data', **kwargs):
        """Add a matrix uncertainty source for use in the fit.

        :param err_matrix: covariance or correlation matrix
        :param matrix_type: One of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param name: Unique name for this uncertainty source. If :py:obj:`None`, the name of the error source will be
                     set to a random alphanumeric string.
        :type name: str or None
        :param err_val: The pointwise uncertainties (mandatory if only a correlation matrix is given).
        :type err_val: typing.Iterable[float]
        :param relative: If :py:obj:`True`, the covariance matrix and/or **err_val** will be interpreted as a *relative*
                         uncertainty.
        :type relative: bool
        :param reference: Either ``'data'`` or ``'model'``. Specifies which reference values to use when calculating
                          absolute errors from relative errors.
        :type reference: str
        :return: An error id which uniquely identifies the created error source.
        :rtype: str
        """
        if reference == 'data':
            # delegate to data container
            _reference_object = self._data_container
        elif reference == 'model':
            # delegate to model container
            _reference_object = self._param_model
            if relative:
                raise NotImplementedError("Errors relative to model not implemented!")
        else:
            raise FitException("Cannot add matrix error: unknown reference "
                               "specification '{}', expected one of: 'data', 'model'...".format(reference))

        _ret = _reference_object.add_matrix_error(err_matrix=err_matrix, matrix_type=matrix_type,
                                                  name=name, err_val=err_val, relative=relative, **kwargs)

        return _ret

    def disable_error(self, err_id):
        """Temporarily disable an uncertainty source so that it doesn't count towards calculating the total uncertainty.

        :param str err_id: error id
        """
        try:
            # try to find error in data container
            _ret = self._data_container.disable_error(err_id)  # TODO: this call does not return anything
        except DataContainerException:
            # try to find error in model container
            _ret = self._param_model.disable_error(err_id)  # TODO: this call does not return anything
        return _ret

    def enable_error(self, err_id):
        """(Re-)Enable an uncertainty source so that it counts towards calculating the total uncertainty.

        :param str err_id: error id
        """
        try:
            # try to find error in data container
            _ret = self._data_container.enable_error(err_id)  # TODO: this call does not return anything
        except DataContainerException:
            # try to find error in model container
            _ret = self._param_model.enable_error(err_id)  # TODO: this call does not return anything
        return _ret

    def do_fit(self, asymmetric_parameter_errors=False):
        """Perform the minimization of the cost function.

        :param bool asymmetric_parameter_errors: If :py:obj:`True`, calculate asymmetric parameter errors.
        :return: A dictionary containing the fit results.
        :rtype: dict
        """
        if self._cost_function.needs_errors and not self.has_errors:
            warnings.warn("Cost function expects errors but no errors were specified.")

        # Give relative model errors data as reference for initial fit:
        self._set_data_as_model_ref()

        # Initial fit:
        self._pre_fit_iteration(first_fit=True)
        self._fitter.do_fit()  # TODO specify other node to minimize
        self._post_fit_iteration(first_fit=True)

        if self._iterative_fits_needed():
            _convergence_limit = float(kc("fit", "iterative_do_fit", "convergence_limit"))
            _previous_cost = self.cost_function_value
            for i in range(kc("fit", "iterative_do_fit", "max_iterations")):
                self._pre_fit_iteration()
                self._fitter.reset_minimizer()  # flush iminuit cache
                self._fitter.do_fit()
                self._post_fit_iteration()
                if abs(self.cost_function_value - _previous_cost) < _convergence_limit:
                    break
                _previous_cost = self.cost_function_value
        elif self._second_fit_needed():
            self._pre_fit_iteration()
            self._fitter.reset_minimizer()  # flush iminuit cache
            self._fitter.do_fit()
            self._post_fit_iteration()

        self._loaded_result_dict = None
        self._update_parameter_formatters()
        return self.get_result_dict(asymmetric_parameter_errors=asymmetric_parameter_errors)

    def assign_model_function_name(self, name):
        """Assign a string to be the model function name.

        :param str name: The new name.
        """
        self._model_function.formatter.name = name

    def assign_model_function_expression(self, expression_format_string):
        """Assign a plain-text-formatted expression string to the model function.

        :param str expression_format_string: The plain text string.
        """
        self._model_function.formatter.expression_format_string = expression_format_string

    def assign_parameter_names(self, **par_names_dict):
        """Assign display strings to all model function arguments.

        :param par_names_dict: Dictionary mapping the parameter names to their display names.
        """
        for _af in self._get_model_function_argument_formatters():
            _an = par_names_dict.pop(_af.arg_name, None)
            if _an is not None:
                _af.name = _an
        if par_names_dict:
            warnings.warn("Could not assign all names to a parameter."
                          "Leftover: {}".format(par_names_dict))

    def assign_model_function_latex_name(self, latex_name):
        """Assign a LaTeX-formatted string to be the model function name.

        :param str latex_name: The LaTeX string.
        """
        self._model_function.formatter.latex_name = latex_name

    def assign_model_function_latex_expression(self, latex_expression_format_string):
        """Assign a LaTeX-formatted expression string to the model function.

        :param str latex_expression_format_string: The LaTeX string. Elements like ``'{par_name}'``
            will be replaced automatically with the corresponding LaTeX names for the given
            parameter. These can be set with :py:meth:`~assign_parameter_latex_names`.
        """
        self._model_function.formatter.latex_expression_format_string = \
            latex_expression_format_string

    def assign_parameter_latex_names(self, **par_latex_names_dict):
        """Assign LaTeX-formatted strings to all model function arguments.

        :param par_latex_names_dict: Dictionary mapping the parameter names to their latex names.
        """
        for _af in self._get_model_function_argument_formatters():
            _aln = par_latex_names_dict.pop(_af.arg_name, None)
            if _aln is not None:
                _af.latex_name = _aln
        if par_latex_names_dict:
            warnings.warn("Could not assign all latex names to a parameter."
                          "Leftover: {}".format(par_latex_names_dict))

    def get_result_dict(self, asymmetric_parameter_errors=False):
        """Return a dictionary of the fit results.

        :param bool asymmetric_parameter_errors: If :py:obj:`True`, calculate asymmetric parameter errors.
        :return: A dictionary containing the fit results.
        :rtype: dict
        """
        _result_dict = dict()

        _result_dict['did_fit'] = self.did_fit
        _cost = float(self.cost_function_value)  # convert numpy scalar to float for yaml representation
        _ndf = self.ndf
        _result_dict['cost'] = _cost
        _result_dict['ndf'] = _ndf
        _gof = self.goodness_of_fit
        _result_dict['goodness_of_fit'] = _gof
        _result_dict['gof/ndf'] = _gof / _ndf if _gof is not None else _gof
        _result_dict['chi2_probability'] = self.chi2_probability
        _result_dict['parameter_values'] = self.parameter_name_value_dict
        if _result_dict['did_fit']:
            _result_dict['parameter_cov_mat'] = self.parameter_cov_mat
            _parameter_errors = OrderedDict()
            for _pn, _pe in zip(self.parameter_names, self.parameter_errors):
                _parameter_errors[_pn] = _pe
            _result_dict['parameter_errors'] = _parameter_errors
            _result_dict['parameter_cor_mat'] = self.parameter_cor_mat
        else:
            _result_dict['parameter_cov_mat'] = None
            _result_dict['parameter_errors'] = None
            _result_dict['parameter_cor_mat'] = None

        if self._loaded_result_dict is not None and self._loaded_result_dict['asymmetric_parameter_errors'] is not None:
            _asymm_errs = self._loaded_result_dict['asymmetric_parameter_errors']
        elif asymmetric_parameter_errors:
            self._check_dynamic_error_compatibility()
            _asymm_errs = self.asymmetric_parameter_errors
        else:
            _asymm_errs = self._fitter.asymmetric_fit_parameter_errors_if_calculated
        if _asymm_errs is not None:
            _asymm_errs_dict = OrderedDict()
            for _pn, _ape in zip(self.parameter_names, _asymm_errs):
                _asymm_errs_dict[_pn] = _ape
            _asymm_errs = _asymm_errs_dict
        _result_dict['asymmetric_parameter_errors'] = _asymm_errs

        return _result_dict

    def report(self, output_stream=sys.stdout, show_data=True, show_model=True, show_fit_results=True,
               asymmetric_parameter_errors=False):
        """Print a summary of the fit state and/or results.

        :param output_stream: The output stream to which the report should be printed.
        :type output_stream: io.TextIOBase
        :param show_data: If :py:obj:`True`, print out information about the data.
        :type show_data: bool
        :param show_model: If :py:obj:`True`, print out information about the parametric model.
        :type show_model: bool
        :param show_fit_results: If :py:obj:`True`, print out information about the fit results.
        :type show_fit_results: bool
        :param asymmetric_parameter_errors: If :py:obj:`True`, use two different parameter errors for up/down
            directions.
        :type asymmetric_parameter_errors: bool
        """

        _indent = ' ' * 4
        if show_data:
            self._report_data(output_stream=output_stream, indent=_indent, indentation_level=0)
        if show_model:
            self._report_model(output_stream=output_stream, indent=_indent, indentation_level=0)
        if show_fit_results:
            self._report_fit_results(output_stream=output_stream, indent=_indent, indentation_level=0,
                                     asymmetric_parameter_errors=asymmetric_parameter_errors)

    def to_file(self, filename, file_format=None, calculate_asymmetric_errors=False):
        """Write kafe2 object to file

        :param filename: Filename for the output.
        :type filename: str
        :param file_format: A format for the output file. If :py:obj:`None`, the extension from the filename is used.
        :type file_format: str or None
        :param calculate_asymmetric_errors: If asymmetric errors should be calculated before saving the results.
        :type calculate_asymmetric_errors: bool
        """
        if calculate_asymmetric_errors:
            _ = self.asymmetric_parameter_errors  # force calculation of asymmetric errors if not calculated yet
        super(FitBase, self).to_file(filename=filename, file_format=file_format)
