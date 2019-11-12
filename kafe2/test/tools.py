import numpy as np
from scipy.optimize import minimize


def calculate_expected_fit_parameters_xy(x_data, y_data, model_function, y_error, initial_parameter_values,
                                         x_error=None, model_function_derivative=None):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    y_error = np.array(y_error)
    initial_parameter_values = np.array(initial_parameter_values)

    def chi2(parameter_values):
        _residuals = y_data - model_function(x_data, *parameter_values)
        return np.sum((_residuals / y_error) ** 2)

    def chi2_xy(parameter_values):
        _residuals = y_data - model_function(x_data, *parameter_values)
        _xy_error_squared = y_error ** 2 + (x_error * model_function_derivative(x_data, *parameter_values)) ** 2
        return np.sum(_residuals ** 2 / _xy_error_squared)

    _result = minimize(fun=chi2, x0=initial_parameter_values)
    if x_error is None:
        return _result.x
    else:
        _result_2 = minimize(fun=chi2_xy, x0=_result.x)
        return _result_2.x

