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
    _result_2 = minimize(fun=chi2_xy, x0=_result.x)
    return _result_2.x


def calculate_expected_multi_xy_chi2_cost(
        y_data_1, y_data_2, y_model_1, y_model_2, y_cov_mat_1, y_cov_mat_2, y_cov_mat_shared,
        x_cov_mat_1, x_cov_mat_2, x_cov_mat_shared, derivatives_1, derivatives_2):

    assert y_data_1.ndim == 1
    _shape_1d = y_data_1.shape
    _middle = _shape_1d[0]
    _end = 2 * _middle
    _shape_2d = 2 * _shape_1d
    assert y_data_2.shape == _shape_1d
    assert y_model_1.shape == _shape_1d
    assert y_model_2.shape == _shape_1d
    assert y_cov_mat_1.shape == _shape_2d
    assert y_cov_mat_2.shape == _shape_2d
    assert y_cov_mat_shared.shape == _shape_2d
    assert x_cov_mat_1.shape == _shape_2d
    assert x_cov_mat_2.shape == _shape_2d
    assert x_cov_mat_shared.shape == _shape_2d
    assert derivatives_1.shape == _shape_1d
    assert derivatives_2.shape == _shape_1d

    _residuals = np.zeros(shape=_end)
    _residuals[0:_middle] = y_data_1 - y_model_1
    _residuals[_middle:_end] = y_data_2 - y_model_2

    _xy_cov_mat = np.zeros(shape=(_end, _end))

    # Add y cov mats first
    _xy_cov_mat[0:_middle, 0:_middle] += y_cov_mat_1
    _xy_cov_mat[0:_middle, 0:_middle] += y_cov_mat_shared
    _xy_cov_mat[_middle:_end, _middle:_end] += y_cov_mat_2
    _xy_cov_mat[_middle:_end, _middle:_end] += y_cov_mat_shared
    _xy_cov_mat[0:_middle, _middle:_end] += y_cov_mat_shared
    _xy_cov_mat[_middle:_end, 0:_middle] += y_cov_mat_shared

    # Add x cov mats afterwards
    _outer_11 = np.outer(derivatives_1, derivatives_1)
    _outer_22 = np.outer(derivatives_2, derivatives_2)
    _outer_12 = np.outer(derivatives_1, derivatives_2)
    _outer_21 = np.outer(derivatives_2, derivatives_1)
    _xy_cov_mat[0:_middle, 0:_middle] += x_cov_mat_1 * _outer_11
    _xy_cov_mat[0:_middle, 0:_middle] += x_cov_mat_shared * _outer_11
    _xy_cov_mat[_middle:_end, _middle:_end] += x_cov_mat_2 * _outer_22
    _xy_cov_mat[_middle:_end, _middle:_end] += x_cov_mat_shared * _outer_22
    _xy_cov_mat[0:_middle, _middle:_end] += x_cov_mat_shared * _outer_12
    _xy_cov_mat[_middle:_end, 0:_middle] += x_cov_mat_shared * _outer_21

    _xy_cov_mat_inv = np.linalg.inv(_xy_cov_mat)

    return np.matmul(_residuals.T, np.matmul(_xy_cov_mat_inv, _residuals))
