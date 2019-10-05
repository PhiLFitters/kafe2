import sys
import numpy as np

from .._base import FitBase
from ...core import CombinedNexus, NexusFitter
from .cost import MultiCostFunction


class MultiFit(FitBase):

    def __init__(self, fit_list, clone_fits=False, minimizer=None, minimizer_kwargs=None):
        if clone_fits:
            raise NotImplementedError()  # TODO implement
        self._underlying_fits = fit_list
        _parameter_names = []
        _nexus_list = []
        _fit_namespaces = []
        for _i, _fit_i in enumerate(self._underlying_fits):
            _nexus_list.append(_fit_i._nexus)
            _fit_namespaces.append('fit%s_' % _i)
            for _parameter_name in _fit_i.parameter_names:
                if _parameter_name not in _parameter_names:
                    _parameter_names.append(_parameter_name)
        self._nexus = CombinedNexus(
            source_nexuses=_nexus_list,
            source_nexus_namespaces=_fit_namespaces,
            shared_parameters=_parameter_names
        )

        _singular_cost_functions = [_underlying_fit._cost_function for _underlying_fit in self._underlying_fits]
        self._cost_function = MultiCostFunction(
            singular_cost_functions=_singular_cost_functions, fit_namespaces=_fit_namespaces)
        self._cost_function.ndf = self.data_size - len(_parameter_names)
        _single_cost_names = ['%scost' % _fit_namespace for _fit_namespace in _fit_namespaces]
        self._nexus.new_function(function_handle=self._cost_function.func, parameter_names=_single_cost_names)
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs
        self._fitter = NexusFitter(
            nexus=self._nexus, parameters_to_fit=_parameter_names, parameter_to_minimize=self._cost_function.name,
            minimizer=self._minimizer, minimizer_kwargs=self._minimizer_kwargs
        )
        self._loaded_result_dict = None

    def assign_parameter_latex_names(self, **par_latex_names_dict):
        for _underlying_fit in self._underlying_fits:
            _underlying_fit.assign_parameter_latex_names(**par_latex_names_dict)

    def assign_model_function_expression(self, expression_format_string, fit_index):
        self._underlying_fits[fit_index].assign_model_function_expression(
            expression_format_string=expression_format_string)

    def assign_model_function_latex_expression(self, latex_expression_format_string, fit_index):
        self._underlying_fits[fit_index].assign_model_function_latex_expression(
            latex_expression_format_string=latex_expression_format_string)

    def _get_model_function_argument_formatters(self):
        _included_argument_names = set()
        _combined_argument_formatters = []
        for _underlying_fit in self._underlying_fits:
            for _argument_formatter in _underlying_fit._get_model_function_argument_formatters():
                if _argument_formatter.name not in _included_argument_names:
                    _combined_argument_formatters.append(_argument_formatter)
                    _included_argument_names.add(_argument_formatter.name)
        return _combined_argument_formatters

    def _invalidate_total_error_cache(self):
        pass

    def _mark_errors_for_update(self):
        pass

    def _get_parameter_indices(self, underlying_fit):
        return [self.parameter_names.index(_underlying_fit_parameter_name)
                for _underlying_fit_parameter_name in underlying_fit.parameter_names]

    def _update_parameter_formatters(self, update_asymmetric_errors=False):
        for _underlying_fit in self._underlying_fits:
            _underlying_fit._update_parameter_formatters(update_asymmetric_errors=update_asymmetric_errors)

    def _update_underlying_fits(self):
        _parameter_name_value_dict = self.parameter_name_value_dict
        for _underlying_fit in self._underlying_fits:
            _parameter_update_dict = {_par_name:_parameter_name_value_dict[_par_name]
                                      for _par_name in _underlying_fit.parameter_names}
            _underlying_fit.set_parameter_values(**_parameter_update_dict)
            _parameter_indices = self._get_parameter_indices(underlying_fit=_underlying_fit)
            _underlying_fit._loaded_result_dict = dict(
                did_fit=self.did_fit,
                parameter_errors=self.parameter_errors[_parameter_indices],
                parameter_cor_mat=self.parameter_cor_mat[_parameter_indices, _parameter_indices],
                parameter_cov_mat=self.parameter_cov_mat[_parameter_indices, _parameter_indices]
            )
        self._update_parameter_formatters()

    def _set_multifit_fitter_for_underlying_fits(self):
        _original_fitters = []
        for _underlying_fit in self._underlying_fits:
            _original_fitters.append(_underlying_fit._fitter)
            _underlying_fit._fitter = self._fitter
        return _original_fitters

    def _restore_original_fitters_for_underlying_fits(self, original_fitters):
        for _original_fitter, _underlying_fit in zip(original_fitters, self._underlying_fits):
            _underlying_fit._fitter = _original_fitter

    @property
    def data(self):
        pass

    @property
    def data_size(self):
        return np.sum([_underlying_fit.data_size for _underlying_fit in self._underlying_fits])

    @property
    def model(self):
        pass

    @property
    def parameter_errors(self):
        _original_fitters = self._set_multifit_fitter_for_underlying_fits()
        _parameter_errors = super(MultiFit, self).parameter_errors
        self._restore_original_fitters_for_underlying_fits(_original_fitters)
        return _parameter_errors

    @property
    def parameter_cor_mat(self):
        _original_fitters = self._set_multifit_fitter_for_underlying_fits()
        _parameter_cor_mat = super(MultiFit, self).parameter_cor_mat
        self._restore_original_fitters_for_underlying_fits(_original_fitters)
        return _parameter_cor_mat

    @property
    def parameter_cov_mat(self):
        _original_fitters = self._set_multifit_fitter_for_underlying_fits()
        _parameter_cov_mat = super(MultiFit, self).parameter_cov_mat
        self._restore_original_fitters_for_underlying_fits(_original_fitters)
        return _parameter_cov_mat

    def do_fit(self):
        """
        Perform the minimization of the cost function.
        """
        for _i, _underlying_fit_i in enumerate(self._underlying_fits):
            if _underlying_fit_i._cost_function.needs_errors and not _underlying_fit_i._data_container.has_errors:
                raise self.EXCEPTION_TYPE('No data errors defined for fit %s' % _i)
        _original_fitters = self._set_multifit_fitter_for_underlying_fits()
        self._fitter.do_fit()
        self._restore_original_fitters_for_underlying_fits(_original_fitters)
        self._update_underlying_fits()
        self._loaded_result_dict = None

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

        for _i, _underlying_fit_i in enumerate(self._underlying_fits):
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
                _underlying_fit_i._report_data(output_stream=output_stream, indent=_indent, indentation_level=1)
            if show_model:
                _underlying_fit_i._report_model(output_stream=output_stream, indent=_indent, indentation_level=1)
        self._report_fit_results(output_stream=output_stream, indent=_indent, indentation_level=0,
                                 asymmetric_parameter_errors=asymmetric_parameter_errors)
