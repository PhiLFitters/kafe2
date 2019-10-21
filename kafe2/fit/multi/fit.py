import sys
import numpy as np

from .._base import FitBase
from ...core import Nexus, NexusFitter
from .cost import MultiCostFunction


class MultiFit(FitBase):

    def __init__(self, fit_list, clone_fits=False, minimizer=None, minimizer_kwargs=None):
        if clone_fits:
            raise NotImplementedError()  # TODO implement
        self._fits = fit_list
        try:
            iter(self._fits)
        except TypeError:
            self._fits = [self._fits]

        _combined_parameter_names = []
        _fit_namespaces = []
        self._nexus = Nexus()
        for _i, _fit_i in enumerate(self._fits):
            _namespace_i = 'fit%s_' % _i
            _fit_namespaces.append(_namespace_i)
            for _parameter_name in _fit_i.parameter_names:
                if _parameter_name not in _combined_parameter_names:
                    _combined_parameter_names.append(_parameter_name)
            self._nexus.source_from(
                other_nexus=_fit_i._nexus, namespace=_namespace_i, namespace_exempt_nodes=_fit_i.parameter_names,
                namespace_exempt_existing_behavior='replace'
            )

        _singular_cost_functions = [_fit._cost_function for _fit in self._fits]
        self._cost_function = MultiCostFunction(
            singular_cost_functions=_singular_cost_functions, fit_namespaces=_fit_namespaces)
        self._cost_function.ndf = self.data_size - len(_combined_parameter_names)
        _single_cost_names = ['%scost' % _fit_namespace for _fit_namespace in _fit_namespaces]
        self._nexus.add_function(func=self._cost_function.func, par_names=_single_cost_names)
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs
        self._fitter = NexusFitter(
            nexus=self._nexus, parameters_to_fit=_combined_parameter_names,
            parameter_to_minimize=self._cost_function.name, minimizer=self._minimizer,
            minimizer_kwargs=self._minimizer_kwargs
        )
        self._fit_param_constraints = []
        self._loaded_result_dict = None

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
            _parameter_update_dict = {_par_name:_parameter_name_value_dict[_par_name]
                                      for _par_name in _fit.parameter_names}
            _fit.set_parameter_values(**_parameter_update_dict)
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

    def _set_new_data(self, new_data):
        raise NotImplementedError()

    def _set_new_parametric_model(self):
        raise NotImplementedError()

    # -- public properties

    @property
    def data(self):
        return [_fit.data for _fit in self._fits]

    @property
    def data_size(self):
        """the size (number of points) of the data container"""
        return np.sum([_fit.data_size for _fit in self._fits])

    @property
    def has_data_errors(self):
        """``True`` if at least one uncertainty source is defined for the data"""
        return np.any([_fit.has_data_errors for _fit in self._fits])

    @property
    def has_model_errors(self):
        """``True`` if at least one uncertainty source is defined for the model"""
        return np.any([_fit.has_model_errors for _fit in self._fits])

    @property
    def model(self):
        return [_fit.model for _fit in self._fits]

    @property
    def model_count(self):
        """the number of model functions contained in the fit, 1 by default"""
        return np.sum([_fit.model_count for _fit in self._fits])

    @property
    def fits(self):
        return self._fits

    # -- public methods

    def add_matrix_error(self, err_matrix, matrix_type, fit_index=None,
                         name=None, err_val=None, relative=False, reference='data', **kwargs):
        if fit_index is None:
            for _fit in self._fits:
                _fit[fit_index].add_matrix_error(
                    err_matrix=err_matrix, matrix_type=matrix_type, name=name, err_val=err_val, relative=relative,
                    reference=reference, **kwargs
                )
        else:
            self._fits[fit_index].add_matrix_error(
                err_matrix=err_matrix, matrix_type=matrix_type, name=name, err_val=err_val, relative=relative,
                reference=reference, **kwargs
            )

    def add_simple_error(self, err_val, fit_index=None, name=None, correlation=0, relative=False, reference='data',
                         **kwargs):
        if fit_index is None:
            for _fit in self._fits:
                _fit[fit_index].add_simple_error(
                    err_val=err_val, name=name, correlation=correlation, relative=relative, reference=reference,
                    **kwargs
                )
        else:
            self._fits[fit_index].add_simple_error(
                err_val=err_val, name=name, correlation=correlation, relative=relative, reference=reference,
                **kwargs
            )

    def assign_model_function_expression(self, expression_format_string, fit_index=None):
        if fit_index is None:
            for _fit in self._fits:
                _fit.assign_model_function_expression(expression_format_string=expression_format_string)
        else:
            self._fits[fit_index].assign_model_function_expression(
                expression_format_string=expression_format_string)

    def assign_model_function_latex_name(self, latex_name, fit_index=None):
        if fit_index is None:
            for _fit in self._fits:
                _fit.assign_model_function_latex_name(latex_name=latex_name)
        else:
            self._fits[fit_index].assign_model_function_latex_name(latex_name=latex_name)

    def assign_model_function_latex_expression(self, latex_expression_format_string, fit_index=None):
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
        :type err_id: int
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
