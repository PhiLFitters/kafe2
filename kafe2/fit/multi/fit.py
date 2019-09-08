from .._base import FitBase
from ...core import CombinedNexus, NexusFitter
from .cost import MultiCostFunction


class MultiFit(FitBase):

    def __init__(self, fit_list, minimizer=None, minimizer_kwargs=None):
        self._underlying_fits = fit_list
        self._parameter_names = []
        _nexus_list = []
        _fit_namespaces = []
        for _i, _fit_i in enumerate(self._underlying_fits):
            _nexus_list.append(_fit_i._nexus)
            _fit_namespaces.append('fit%s_' % _i)
            for _parameter_name in _fit_i.parameter_names:
                if _parameter_name not in self._parameter_names:
                    self._parameter_names.append(_parameter_name)
        self._nexus = CombinedNexus(
            source_nexuses=_nexus_list,
            source_nexus_namespaces=_fit_namespaces,
            shared_parameters=self._parameter_names
        )

        self._cost_function = MultiCostFunction()
        _single_cost_names = ['%scost' % _fit_namespace for _fit_namespace in _fit_namespaces]
        self._nexus.new_function(function_handle=self._cost_function.func, parameter_names=_single_cost_names)
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs
        self._fitter = NexusFitter(
            nexus=self._nexus, parameters_to_fit=self._parameter_names, parameter_to_minimize=self._cost_function.name,
            minimizer=self._minimizer, minimizer_kwargs=self._minimizer_kwargs
        )

    def assign_parameter_latex_names(self, **par_latex_names_dict):
        for _underlying_fit in self._underlying_fits:
            _underlying_fit.assign_parameter_latex_names(**par_latex_names_dict)

    def assign_model_function_expression(self, expression_format_string, fit_index):
        self._underlying_fits[fit_index].assign_model_function_expression(
            expression_format_string=expression_format_string)

    def assign_model_function_latex_expression(self, latex_expression_format_string, fit_index):
        self._underlying_fits[fit_index].assign_model_function_latex_expression(
            latex_expression_format_string=latex_expression_format_string)

    def _invalidate_total_error_cache(self):
        pass

    def _mark_errors_for_update(self):
        pass

    def _update_parameter_formatters(self, update_asymmetric_errors=False):
        pass

    def data(self):
        pass

    def model(self):
        pass

    def do_fit(self):
        """
        Perform the minimization of the cost function.
        """
        for _i, _underlying_fit_i in enumerate(self._underlying_fits):
            if _underlying_fit_i._cost_function.needs_errors and not _underlying_fit_i._data_container.has_errors:
                raise self.EXCEPTION_TYPE('No data errors defined for fit %s' % _i)
        _original_fitters = []
        _original_minimizers = []
        for _underlying_fit in self._underlying_fits:
            _original_fitters.append(_underlying_fit._fitter)
            _original_minimizers.append(_underlying_fit._minimizer)
            _underlying_fit._fitter = self._fitter
        self._fitter.do_fit()
        for _original_fitter, _original_minimizer, _underlying_fit in zip(
                _original_fitters, _original_minimizers, self._underlying_fits):
            _underlying_fit._fitter = _original_fitter
            _underlying_fit._minimizer = _original_minimizer
        self._loaded_result_dict = None
        self._update_parameter_formatters()
