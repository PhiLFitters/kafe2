import six
import sys
from .._base import FitBase, CostFunction, ModelFunctionBase
from ...core.fitters.nexus import Parameter, Array


class CustomFit(FitBase):
    _MODEL_ERROR_NODE_NAMES = []

    def __init__(
            self, cost_function, minimizer=None, minimizer_kwargs=None):
        if not isinstance(cost_function, CostFunction):
            cost_function = CostFunction(
                cost_function, add_constraint_cost=True, add_determinant_cost=False)
            cost_function._needs_errors = False
        super().__init__(
            data=None, model_function=None, cost_function=cost_function,
            minimizer=minimizer, minimizer_kwargs=minimizer_kwargs,
            dynamic_error_algorithm="nonlinear"
        )
        _dummy_model_function = ModelFunctionBase(cost_function.func, independent_argcount=0)
        _parameter_nodes = []
        for _par_name, _par_value in six.iteritems(_dummy_model_function.defaults_dict):
            _parameter_nodes.append(self._nexus.add(Parameter(_par_value, name=_par_name),
                                                    existing_behavior="replace"))
            self._fit_param_names.append(_par_name)
        self._nexus.add(Array(_parameter_nodes, name="parameter_values"),
                        existing_behavior="replace")
        self._parameter_formatters = _dummy_model_function.formatter.arg_formatters
        self._initialize_fitter()
        """
        Construct a fit without explicit data and model from only a cost function.
        
        :param cost_function: the function to minimize.
        :type cost_function: callable
        :param minimizer: Name of the minimizer to use.
        :type minimizer: str or None
        :param minimizer_kwargs: Dictionary wit keywords for initializing the minimizer.
        :type minimizer_kwargs: dict or None
        """

    def _set_new_data(self, new_data):
        pass

    def _set_new_parametric_model(self):
        pass

    def _set_data_as_model_ref(self):
        pass

    def _get_model_function_argument_formatters(self):
        return self._parameter_formatters

    def _get_model_function_parameter_formatters(self):
        return self._parameter_formatters

    def _iterative_fits_needed(self):
        return False

    def _second_fit_needed(self):
        return False

    @property
    def data(self):
        return None

    @property
    def data_error(self):
        return None

    @property
    def data_cov_mat(self):
        return None

    @property
    def data_cov_mat_inverse(self):
        return None

    @property
    def data_cor_mat(self):
        return None

    @property
    def model(self):
        return None

    @property
    def model_error(self):
        return None

    @property
    def model_cov_mat(self):
        return None

    @property
    def model_cov_mat_inverse(self):
        return None

    @property
    def model_cor_mat(self):
        return None

    @property
    def total_error(self):
        return None

    @property
    def total_cov_mat(self):
        return None

    @property
    def total_cov_mat_inverse(self):
        return None

    @property
    def total_cor_mat(self):
        return None

    @property
    def model_label(self):
        return None

    @property
    def data_size(self):
        return None

    @property
    def model_count(self):
        return 0

    @property
    def ndf(self):
        return None

    @property
    def goodness_of_fit(self):
        return None

    @property
    def chi2_probability(self):
        return None

    def report(self, output_stream=sys.stdout, asymmetric_parameter_errors=False):
        super().report(
            output_stream=output_stream, show_data=False, show_model=False, show_fit_results=True,
            asymmetric_parameter_errors=asymmetric_parameter_errors)