import sys
from copy import copy
import numpy as np

from .._base import FitBase
from ...core import Nexus, NexusFitter
from ...core.error import SimpleGaussianError, MatrixGaussianError
from ...core.fitters.nexus import Alias, Function, Array
from ...tools import random_alphanumeric
from .cost import MultiCostFunction, SharedChi2CostFunction
from collections import OrderedDict


class MultiFit(FitBase):
    """
    A MultiFit combines several regular fits into a combined fit object. Calling do_fit on the
    MultiFit will result in a numerical minimization of the sum of the cost functions of the
    individual fits. Parameters with the same name will be unified: their value will be the same
    across all individual fits.
    """
    def __init__(self, fit_list, minimizer=None, minimizer_kwargs=None):
        """
        :param fit_list: the individual fits from which to create the MultiFit.
        :type fit_list: iterable of FitBase
        :param minimizer: the minimizer to use for fitting.
        :type minimizer: None, "iminuit", "tminuit", or "scipy".
        :param minimizer_kwargs: dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """
        self._fits = fit_list
        try:
            iter(self._fits)
        except TypeError:
            self._fits = [self._fits]

        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs
        self._shared_error_dicts = dict()
        self._min_x_error = None
        self._init_nexus_callbacks = []
        self._init_nexus()
        self._initialize_fitter()

        self._fit_param_constraints = []
        self._loaded_result_dict = None

        for _fit in self._fits:
            _fit._init_nexus_callbacks.append(self._init_nexus)

    # -- private methods

    def _get_model_function_argument_formatters(self):
        _included_argument_names = set()
        _combined_argument_formatters = []
        for _fit in self._fits:
            for _argument_formatter in _fit._get_model_function_argument_formatters():
                if _argument_formatter.name not in _included_argument_names:
                    _combined_argument_formatters.append(_argument_formatter)
                    _included_argument_names.add(_argument_formatter.name)
        return _combined_argument_formatters

    def _invalidate_total_error_cache(self):
        pass

    def _mark_errors_for_update(self):
        pass

    def _get_parameter_indices(self, singular_fit):
        return [self.parameter_names.index(_parameter_name) for _parameter_name in singular_fit.parameter_names]

    def _update_parameter_formatters(self, update_asymmetric_errors=False):
        for _fit in self._fits:
            _fit._update_parameter_formatters(update_asymmetric_errors=update_asymmetric_errors)

    def _update_singular_fits(self):
        _parameter_name_value_dict = self.parameter_name_value_dict
        for _fit in self._fits:
            _parameter_indices = self._get_parameter_indices(singular_fit=_fit)
            _asymmetric_parameter_errors = self._fitter.asymmetric_fit_parameter_errors_if_calculated
            if _asymmetric_parameter_errors is not None:
                _asymmetric_parameter_errors = _asymmetric_parameter_errors[_parameter_indices]
            _par_cor_mat = self.parameter_cor_mat
            if _par_cor_mat is not None:
                _par_cor_mat = _par_cor_mat[_parameter_indices][:, _parameter_indices]
            _par_cov_mat = self.parameter_cov_mat
            if _par_cov_mat is not None:
                _par_cov_mat = _par_cov_mat[_parameter_indices][:, _parameter_indices]
            _fit._loaded_result_dict = dict(
                did_fit=self.did_fit,
                parameter_errors=self.parameter_errors[_parameter_indices],
                parameter_cor_mat=_par_cor_mat,
                parameter_cov_mat=_par_cov_mat,
                asymmetric_parameter_errors=_asymmetric_parameter_errors
            )
        self._update_parameter_formatters()

    def _init_nexus(self):
        self._nexus = Nexus()
        self._nexus.add_function(
            lambda: self.parameter_constraints, func_name='parameter_constraints')
        self._combined_parameter_node_dict = OrderedDict()
        _cost_names = []
        for _i, _fit_i in enumerate(self._fits):
            _original_cost_i = _fit_i._nexus.get('cost')
            _cost_alias_name_i = 'cost%s' % _i
            _cost_alias_i = Alias(ref=_original_cost_i, name=_cost_alias_name_i)
            self._nexus.add(_cost_alias_i, add_children=False)
            _cost_names.append(_cost_alias_name_i)
            for _par_node in _fit_i.poi_names:
                self._combined_parameter_node_dict[_par_node] = _fit_i._nexus.get(_par_node)
        for _par_node in self._combined_parameter_node_dict.values():
            self._nexus.add(_par_node)
            for _fit in self._fits:
                if _par_node.name in _fit.poi_names:
                    _fit._nexus.add(node=_par_node, existing_behavior='replace')
        self._nexus.add(
            Array(nodes=self._combined_parameter_node_dict.values(), name='parameter_values'))

        _x_data_names = []
        _y_data_names = []
        for _i, _fit_i in enumerate(self._fits):
            _fit_i._initialize_fitter()

            _x_data_node = _fit_i._nexus.get('x_data')
            if _x_data_node is not None:
                _x_data_name = 'x_data%s' % _i
                self._nexus.add(
                    Alias(ref=_x_data_node, name=_x_data_name),
                    add_children=False)
                _x_data_names.append(_x_data_name)

            _y_data_node = _fit_i._nexus.get('y_data')
            if _y_data_node is not None:
                _y_data_name = 'y_data%s' % _i
                self._nexus.add(
                    Alias(ref=_y_data_node, name=_y_data_name),
                    add_children=False)
                _y_data_names.append(_y_data_name)

        if not self._shared_error_dicts:
            _cost_functions = [_fit._cost_function for _fit in self._fits]
        else:
            _chi2_indices = []  # Indices of fits with chi2 cost function
            _data_indices = [0]  # Indices of edges of covariance block in combined matrix
            _cost_functions = []
            _cost_names = []
            _x_cov_mat_names = []
            _derivative_names = []
            _y_model_names = []
            _y_cov_mat_names = []
            _has_shared_x_error = False
            for _err_dict in self._shared_error_dicts.values():
                if _err_dict['axis'] == 'x':
                    _has_shared_x_error = True
                    break
            for _i, _fit_i in enumerate(self._fits):
                if _fit_i._cost_function.is_chi2:
                    _chi2_indices.append(_i)
                    _data_indices.append(_data_indices[-1] + _fit_i.data_size)

                    if _has_shared_x_error:
                        _x_cov_mat_name = 'x_cov_mat%s' % _i
                        self._nexus.add(
                            Alias(ref=_fit_i._nexus.get('x_total_cov_mat'), name=_x_cov_mat_name),
                            add_children=False)
                        _x_cov_mat_names.append(_x_cov_mat_name)

                        _derivatives_name = 'derivatives%s' % _i

                        # Bind fit object (Python is call-by-value), otherwise all derivatives would
                        # use the same fit.
                        def _get_derivatives_func(fit):
                            return lambda: fit._param_model.eval_model_function_derivative_by_x(
                                model_parameters=fit.poi_values,
                                dx=0.01*self._min_x_error
                            )

                        self._nexus.add(
                            Function(func=_get_derivatives_func(_fit_i), name=_derivatives_name),
                            add_children=False)
                        self._nexus.add_dependency(
                            name=_derivatives_name, depends_on='parameter_values')
                        _derivative_names.append(_derivatives_name)

                    _y_model_name = 'y_model%s' % _i
                    self._nexus.add(
                        Alias(ref=_fit_i._nexus.get('y_model'), name=_y_model_name),
                        add_children=False)
                    _y_model_names.append(_y_model_name)

                    _y_cov_mat_name = 'y_cov_mat%s' % _i
                    self._nexus.add(
                        Alias(ref=_fit_i._nexus.get('y_total_cov_mat'), name=_y_cov_mat_name),
                        add_children=False)
                    _y_cov_mat_names.append(_y_cov_mat_name)
                else:
                    _cost_functions.append(_fit_i._cost_function)
                    _cost_names.append('cost%s' % _i)

            def _combine_1d(*single_fit_properties):
                _combined_property = np.zeros(shape=_data_indices[-1])
                for _j, (_single_fit_property, _chi2_index) in enumerate(
                        zip(single_fit_properties, _chi2_indices)):
                    _lower = _data_indices[_j]
                    _upper = _data_indices[_j + 1]
                    _combined_property[_lower:_upper] = _single_fit_property
                return _combined_property

            def _combine_2d(axis_name, *single_fit_properties):
                _combined_property = np.zeros(shape=(_data_indices[-1], _data_indices[-1]))
                for _j, (_single_fit_property, _chi2_index) in enumerate(
                        zip(single_fit_properties, _chi2_indices)):
                    _lower = _data_indices[_j]
                    _upper = _data_indices[_j + 1]
                    _combined_property[_lower:_upper, _lower:_upper] = _single_fit_property
                for _error_dict in self._shared_error_dicts.values():
                    if _error_dict['axis'] == axis_name:
                        _error = _error_dict['err']
                        for _j in range(len(_error.fit_indices)):
                            _lower_1 = _data_indices[_j]
                            _upper_1 = _data_indices[_j + 1]
                            _combined_property[
                                    _lower_1:_upper_1, _lower_1:_upper_1] += _error.cov_mat
                            for _k in range(_j):
                                _lower_2 = _data_indices[_k]
                                _upper_2 = _data_indices[_k + 1]
                                _combined_property[
                                        _lower_1:_upper_1, _lower_2:_upper_2] += _error.cov_mat
                                _combined_property[
                                        _lower_2:_upper_2, _lower_1:_upper_1] += _error.cov_mat
                return _combined_property

            if _has_shared_x_error:
                self._nexus.add_function(
                    func=lambda *p: _combine_2d('x', *p),
                    func_name='x_cov_mat', par_names=_x_cov_mat_names, add_children=False)
                self._nexus.add_function(
                    func=_combine_1d, func_name='derivatives', par_names=_derivative_names,
                    add_children=False)

            self._nexus.add_function(
                func=_combine_1d, func_name='y_data', par_names=_y_data_names, add_children=False)
            self._nexus.add_alias(name='data', alias_for='y_data')
            self._nexus.add_function(
                func=_combine_1d, func_name='y_model', par_names=_y_model_names, add_children=False)
            self._nexus.add_alias(name='model', alias_for='y_model')
            self._nexus.add_function(
                func=lambda *p: _combine_2d('y', *p),
                func_name='y_cov_mat', par_names=_y_cov_mat_names, add_children=False)

            if _has_shared_x_error:
                def total_cov_mat_inverse(x_cov_mat, derivatives, y_cov_mat):
                    _cov_mat = y_cov_mat + x_cov_mat * np.outer(derivatives, derivatives)
                    return np.linalg.inv(_cov_mat)
            else:
                def total_cov_mat_inverse(y_cov_mat):
                    return np.linalg.inv(y_cov_mat)
            self._nexus.add_function(total_cov_mat_inverse)
            _shared_chi2_cost_function = SharedChi2CostFunction()
            self._nexus.add_function(func=_shared_chi2_cost_function.func, func_name='cost_shared')
            _cost_functions.append(_shared_chi2_cost_function)
            _cost_names.append('cost_shared')

        self._cost_function = MultiCostFunction(singular_cost_functions=_cost_functions)
        self._cost_function.ndf = self.data_size - len(self._combined_parameter_node_dict.keys())
        _cost_function_node = self._nexus.add_function(
            func=self._cost_function.func, par_names=_cost_names)
        self._nexus.add_alias(name='cost', alias_for=_cost_function_node.name)
        self._initialize_fitter()
        for _callback in self._init_nexus_callbacks:
            _callback()

    def _initialize_fitter(self):
        self._fitter = NexusFitter(
            nexus=self._nexus, parameters_to_fit=list(self._combined_parameter_node_dict.keys()),
            parameter_to_minimize=self._cost_function.name, minimizer=self._minimizer,
            minimizer_kwargs=self._minimizer_kwargs
        )

    def _add_error_object(self, error_object, reference, name=None, axis=None):
        from ..indexed import IndexedFit
        from ..xy import XYFit

        if axis not in [None, 'x', 'y']:
            raise ValueError("axis must be one of: None, 'x', 'y'")
        _data_size_0 = self._fits[error_object.fit_indices[0]].data_size
        for _fit_index in error_object.fit_indices:
            if not self._fits[_fit_index]._cost_function.is_chi2:
                raise ValueError(
                    "Cannot add shared error because cost function of fit %s is not chi2!"
                    % _fit_index)
            if isinstance(self._fits[_fit_index], XYFit) and axis is None:
                raise ValueError("axis=None is ambiguous for fit %s because it is an XYFit!" % axis)
            if isinstance(self._fits[_fit_index], IndexedFit) and axis == 'x':
                raise ValueError(
                    "axis='x' is incompatible with fit %s because it is an IndexedFit!")
            if self._fits[_fit_index].data_size != _data_size_0:
                raise ValueError("Fit %s data_size not the same as data_size of Fit 0!")
        if name in self._shared_error_dicts:
            raise ValueError("Error with name=%s already exists!" % name)
        name = random_alphanumeric(8)
        while name in self._shared_error_dicts:
            name = random_alphanumeric(8)

        if error_object.relative:
            if not reference == 'data' or reference == 'model':
                raise ValueError(
                    "Error reference must be either 'model' or 'data' but received %s" % reference)
            if axis == 'x' and reference == 'data':
                _node_name = 'x_data%s'
            elif axis == 'y' and reference == 'data':
                _node_name = 'y_data%s'
            elif axis == 'x' and reference == 'model':
                _node_name = 'x_model%s'
                raise NotImplementedError("Errors relative to model not implemented")
            elif axis == 'y' and reference == 'model':
                _node_name = 'y_model%s'
                raise NotImplementedError("Errors relative to model not implemented")
            else:
                raise AssertionError()
            _reference_value = None
            _first_index = None
            for _fit_index in error_object.fit_indices:
                if _reference_value is None:
                    _reference_value = self._nexus.get(_node_name % _fit_index).value
                    _first_index = _fit_index
                else:
                    _other_reference_value = self._nexus.get(_node_name % _fit_index).value
                    if _reference_value.shape != _other_reference_value.shape or np.any(
                            (_reference_value - _other_reference_value) != 0):
                        raise ValueError(
                            "Cannot add a relative error for fits %s and %s because they have "
                            "conflicting references." % (_first_index, _fit_index))
            error_object.reference = _reference_value

        if axis is None:
            axis = 'y'
        elif axis == 'x':
            _min_of_new_error = np.min(error_object.error)
            if self._min_x_error is None:
                self._min_x_error = _min_of_new_error
            else:
                self._min_x_error = min(
                    self._min_x_error, _min_of_new_error
                )

        _error_dict = dict(err=error_object, enabled=True, axis=axis, reference_name=reference)
        self._shared_error_dicts[name] = _error_dict
        self._init_nexus()
        return name

    def _set_new_data(self, new_data):
        raise NotImplementedError()

    def _set_new_parametric_model(self):
        raise NotImplementedError()

    # -- public properties

    @property
    def data(self):
        """list of the data of the individual fits."""
        return [_fit.data for _fit in self._fits]

    @property
    def data_size(self):
        """combined size of the data containers of the individual fits"""
        return np.sum([_fit.data_size for _fit in self._fits])

    @property
    def has_data_errors(self):
        """``True`` if at least one uncertainty source is defined for the data of any of the
        individual fits."""
        return np.any([_fit.has_data_errors for _fit in self._fits])

    @property
    def has_model_errors(self):
        """``True`` if at least one uncertainty source is defined for the model of any of the
        individual fits"""
        return np.any([_fit.has_model_errors for _fit in self._fits])

    @property
    def model(self):
        """list of the model values of the individual fits."""
        return [_fit.model for _fit in self._fits]

    @property
    def model_count(self):
        """the number of model functions contained in the fit, 1 by default"""
        return np.sum([_fit.model_count for _fit in self._fits])

    @property
    def fits(self):
        """the iterable of individual fits that the MultiFit is based on"""
        return copy(self._fits)  # shallow copy

    @property
    def asymmetric_parameter_errors(self):
        """the current asymmetric parameter uncertainties"""
        _asymm_par_errs = super(MultiFit, self).asymmetric_parameter_errors
        self._update_singular_fits()
        return _asymm_par_errs

    # -- public methods

    def add_matrix_error(self, err_matrix, matrix_type, fits, axis=None, name=None, err_val=None,
                         relative=False, reference='data', **kwargs):
        """
        Add a matrix uncertainty source for use in the fit.
        Returns an error id which uniquely identifies the created error source.

        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param fits: the indices of the fits to add the error. If "all", adds the error to all fits.
        :type fits: iterable of int or ``'all'``.
        :param axis: axis of the individual fits to add the error to. ``y``/``1`` errors are treated
        as regular errors for IndexedFits.
        :type axis: int or str
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is
        given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted
        as a *relative* uncertainty
        :type relative: bool
        :param reference: which reference values to use when calculating absolute errors from
        relative errors
        :type reference: 'data' or 'model'
        :return: error id
        :rtype: str
        """
        # TODO relative errors
        if isinstance(fits, int):
            self._fits[fits].add_matrix_error(
                err_matrix=err_matrix, matrix_type=matrix_type, name=name, err_val=err_val,
                axis=axis, reference=reference, relative=relative, **kwargs
            )
        else:
            if fits == 'all':
                fits = list(range(len(self._fits)))
            _matrix_error = MatrixGaussianError(
                err_matrix=err_matrix, matrix_type=matrix_type, err_val=err_val, relative=relative,
                fit_indices=fits
            )
            return self._add_error_object(error_object=_matrix_error, reference=reference,
                                          name=name, axis=axis)

    def add_simple_error(
            self, err_val, fits, axis=None, name=None, correlation=0, relative=False,
            reference='data', **kwargs):
        """
        Add a simple uncertainty source to the fit.
        Returns an error id which uniquely identifies the created error source.

        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param fits: the indices of the fits to add the error. If "all", adds the error to all fits.
        :type fits: iterable of int or ``'all'``.
        :param axis: axis of the individual fits to add the error to. ``y``/``1`` errors are treated
        as regular errors for IndexedFits.
        :type axis: int or str
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :param reference: which reference values to use when calculating absolute errors from relative errors
        :type reference: 'data' or 'model'
        :return: error id
        :rtype: str
        """
        if isinstance(fits, int):
            self._fits[fits].add_simple_error(
                err_val=err_val, name=name, correlation=correlation, relative=relative,
                axis=axis, reference=reference, **kwargs
            )
        else:
            if fits == 'all':
                fits = list(range(len(self._fits)))
            try:
                err_val.ndim  # will raise if simple float
            except AttributeError:
                err_val = np.asarray(err_val, dtype=float)

            if err_val.ndim == 0:  # if dimensionless numpy array (i.e. float64), add a dimension
                err_val = np.ones(self._fits[fits[0]].data_size) * err_val
                # Consistent data size is checked later.
            _simple_error = SimpleGaussianError(
                err_val=err_val, corr_coeff=correlation, relative=relative, fit_indices=fits
            )
            return self._add_error_object(error_object=_simple_error, reference=reference,
                                          name=name, axis=axis)

    def assign_model_function_expression(self, expression_format_string, fit_index=None):
        """Assign a plain-text-formatted expression string to the model function of one of the
        individual fits."""
        if fit_index is None:
            for _fit in self._fits:
                _fit.assign_model_function_expression(
                    expression_format_string=expression_format_string)
        else:
            self._fits[fit_index].assign_model_function_expression(
                expression_format_string=expression_format_string)

    def assign_model_function_latex_name(self, latex_name, fit_index=None):
        """Assign a LaTeX-formatted string to be the model function name of one of the
        individual fits."""
        if fit_index is None:
            for _fit in self._fits:
                _fit.assign_model_function_latex_name(latex_name=latex_name)
        else:
            self._fits[fit_index].assign_model_function_latex_name(latex_name=latex_name)

    def assign_model_function_latex_expression(self, latex_expression_format_string, fit_index=None):
        """Assign a LaTeX-formatted expression string to the model function of one of the individual
        fits."""
        if fit_index is None:
            for _fit in self._fits:
                _fit.assign_model_function_latex_expression(
                    latex_expression_format_string=latex_expression_format_string)
        else:
            self._fits[fit_index].assign_model_function_latex_expression(
                latex_expression_format_string=latex_expression_format_string)

    def assign_parameter_latex_names(self, **par_latex_names_dict):
        for _fit in self._fits:
            _fit.assign_parameter_latex_names(**par_latex_names_dict)

    def disable_error(self, err_id):
        """
        Temporarily disable an uncertainty source so that it doesn't count towards calculating the
        total uncertainty.

        :param err_id: error id
        :type err_id: str
        """
        for _fit in self._fits:
            _fit.disable_error(err_id=err_id)

    def do_fit(self):
        """
        Perform the minimization of the cost function.
        """
        for _i, _fit_i in enumerate(self._fits):
            if _fit_i._cost_function.needs_errors and not _fit_i._data_container.has_errors:
                raise self.EXCEPTION_TYPE('No data errors defined for fit %s' % _i)
        self._fitter.do_fit()
        self._update_singular_fits()
        self._loaded_result_dict = None

    def get_matching_errors(self, fit_index=None, matching_criteria=None, matching_type='equal'):
        if fit_index is None:
            _combined_matching_errors = dict()
            for _fit in self._fits:
                _matching_errors = _fit.get_matching_errors(
                    matching_criteria=matching_criteria, matching_type=matching_type)
                for _key in _matching_errors:
                    # FATAL: there is an error with the same name in separate fits
                    assert _key not in _combined_matching_errors
                    _combined_matching_errors[_key] = _matching_errors[_key]
            return _combined_matching_errors
        else:
            return self._fits[fit_index].get_matching_errors(
                matching_criteria=matching_criteria, matching_type=matching_type)

    def report(self, output_stream=sys.stdout, show_data=True, show_model=True, show_fit_results=True,
               asymmetric_parameter_errors=False):
        """
        Print a summary of the fit state and/or results.

        :param output_stream: the output stream to which the report should be printed
        :type output_stream: TextIOBase
        :param show_data: if ``True``, print out information about the data
        :type show_data: bool
        :param show_model: if ``True``, print out information about the parametric model
        :type show_model: bool
        :param show_fit_results: if ``True``, print out information about the fit results
        :type show_fit_results: bool
        :param asymmetric_parameter_errors: if ``True``, use two different parameter errors for up/down directions
        :type asymmetric_parameter_errors: bool
        """
        _indent = ' ' * 4

        for _i, _fit in enumerate(self._fits):
            _header_string = '#########\n'
            if _i > 9:
                _header_string = '#' + _header_string
            if _i > 99:
                _header_string = '#' + _header_string
            output_stream.write(_header_string)
            output_stream.write('# Fit %s #\n' % _i)
            output_stream.write(_header_string)
            output_stream.write('\n')

            if show_data:
                _fit._report_data(output_stream=output_stream, indent=_indent, indentation_level=1)
            if show_model:
                _fit._report_model(output_stream=output_stream, indent=_indent, indentation_level=1)
        self._report_fit_results(output_stream=output_stream, indent=_indent, indentation_level=0,
                                 asymmetric_parameter_errors=asymmetric_parameter_errors)
