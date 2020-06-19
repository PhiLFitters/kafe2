import unittest2 as unittest
import numpy as np

from kafe2.fit import HistContainer, HistFit, IndexedContainer, IndexedFit, XYContainer, XYFit

from kafe2.fit.indexed.fit import IndexedFitException
from kafe2.fit.xy.fit import XYFitException
from kafe2.fit.histogram.fit import HistFitException


class TestParameterConstraintInHistFit(unittest.TestCase):

    @staticmethod
    def _model_function(x, a, b):
        return a * x + b

    @staticmethod
    def _model_function_antiderivative(x, a, b):
        return 0.5 * a * x ** 2 + b * x

    def _expected_profile_diff(self, res, cov_mat_inv):
        return res.dot(cov_mat_inv).dot(res)

    def _test_consistency(self, constrained_fit, par_cov_mat_inv):
        constrained_fit.do_fit()
        _cost_function = constrained_fit._fitter._fcn_wrapper
        for _i in range(4):
            for _j in range(9):
                _a = self._test_par_values[_i, 0, _j]
                _b = self._test_par_values[_i, 1, _j]
                _profile_constrained = _cost_function(self._test_par_values[_i, 0, _j], self._test_par_values[_i, 1, _j])
                _diff = _profile_constrained - self._profile_no_constraints[_i, _j]
                _expected_profile_diff = self._expected_profile_diff(self._test_par_res[_i, _j], par_cov_mat_inv)
                self.assertTrue(np.abs(_diff - _expected_profile_diff) < 1e-12)

    def setUp(self):
        _bin_edges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        _data = [
            0.5,
            1.5, 1.5,
            2.5, 2.5, 2.5,
            3.5, 3.5, 3.5, 3.5,
            4.5, 4.5, 4.5, 4.5, 4.5
        ]
        self._means = np.array([3.654, 7.789])
        self._vars = np.array([2.467, 1.543])
        self._cov_mat_uncor = np.array([[self._vars[0], 0.0], [0.0, self._vars[1]]])
        self._cov_mat_uncor_inv = np.linalg.inv(self._cov_mat_uncor)
        self._cov_mat_cor = np.array([[self._vars[0], 0.1], [0.1, self._vars[1]]])
        self._cov_mat_cor_inv = np.linalg.inv(self._cov_mat_cor)
        self._cov_mat_simple_a_inv = np.array([[1.0 / self._vars[0], 0.0], [0.0, 0.0]])
        self._cov_mat_simple_b_inv = np.array([[0.0, 0.0], [0.0, 1.0 / self._vars[1]]])

        self._data_container = HistContainer(n_bins=5, bin_range=(0.0, 5.0), fill_data=_data, dtype=float)
        self._data_container.add_error(err_val=1.0)

        _a_test = np.linspace(start=1, stop=2, num=9, endpoint=True)
        _b_test = np.linspace(start=2, stop=3, num=9, endpoint=True)
        self._test_par_values = np.zeros((4, 2, 9))
        self._test_par_values[0, 0] = _a_test
        self._test_par_values[1, 1] = _b_test
        self._test_par_values[2, 0] = _a_test
        self._test_par_values[2, 1] = _b_test
        self._test_par_values[3, 0] = _a_test
        self._test_par_values[3, 1] = _b_test[::-1]  # reverse order
        self._test_par_res = self._test_par_values - self._means.reshape((1, 2, 1))
        self._test_par_res = np.transpose(self._test_par_res, axes=(0, 2, 1))

        self._fit_no_constraints = HistFit(self._data_container, model_density_function=self._model_function,
                                           bin_evaluation=self._model_function_antiderivative)
        self._fit_no_constraints.do_fit()
        _cost_function = self._fit_no_constraints._fitter._fcn_wrapper
        self._profile_no_constraints = np.zeros((4, 9))
        for _i in range(4):
            for _j in range(9):
                self._profile_no_constraints[_i, _j] = _cost_function(
                    self._test_par_values[_i, 0, _j],
                    self._test_par_values[_i, 1, _j])

    def test_bad_input_exception(self):
        _fit_with_constraint = HistFit(self._data_container, model_density_function=self._model_function,
                                       bin_evaluation=self._model_function_antiderivative)
        with self.assertRaises(HistFitException):
            _fit_with_constraint.add_parameter_constraint('c', 1.0, 1.0)
        with self.assertRaises(HistFitException):
            _fit_with_constraint.add_matrix_parameter_constraint(['a', 'c'], [1.0, 2.0], [[0.2, 0.0], [0.0, 0.1]])
        with self.assertRaises(HistFitException):
            _fit_with_constraint.add_matrix_parameter_constraint(['a'], [1.0, 2.0], [[0.2, 0.0], [0.0, 0.1]])

    def test_fit_profile_cov_mat_uncorrelated(self):
        _fit_with_constraint = HistFit(self._data_container, model_density_function=self._model_function,
                                       bin_evaluation=self._model_function_antiderivative)
        _fit_with_constraint.add_matrix_parameter_constraint(['a', 'b'], self._means, self._cov_mat_uncor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_uncor_inv)
        _fit_with_constraint_alt = HistFit(self._data_container, model_density_function=self._model_function,
                                           bin_evaluation=self._model_function_antiderivative)
        _fit_with_constraint_alt.add_parameter_constraint('a', self._means[0], np.sqrt(self._vars[0]))
        _fit_with_constraint_alt.add_parameter_constraint('b', self._means[1], np.sqrt(self._vars[1]))
        self._test_consistency(_fit_with_constraint_alt, self._cov_mat_uncor_inv)

    def test_fit_profile_cov_mat_correlated(self):
        _fit_with_constraint = HistFit(self._data_container, model_density_function=self._model_function,
                                       bin_evaluation=self._model_function_antiderivative)
        _fit_with_constraint.add_matrix_parameter_constraint(['a', 'b'], self._means, self._cov_mat_cor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_cor_inv)

    def test_fit_profile_simple_a(self):
        _fit_with_constraint = HistFit(self._data_container, model_density_function=self._model_function,
                                       bin_evaluation=self._model_function_antiderivative)
        _fit_with_constraint.add_parameter_constraint('a', self._means[0], np.sqrt(self._vars[0]))
        self._test_consistency(_fit_with_constraint, self._cov_mat_simple_a_inv)

    def test_fit_profile_simple_b(self):
        _fit_with_constraint = HistFit(self._data_container, model_density_function=self._model_function,
                                       bin_evaluation=self._model_function_antiderivative)
        _fit_with_constraint.add_parameter_constraint('b', self._means[1], np.sqrt(self._vars[1]))
        self._test_consistency(_fit_with_constraint, self._cov_mat_simple_b_inv)


class TestParameterConstraintInIndexedFit(unittest.TestCase):

    def _expected_profile_diff(self, res, cov_mat_inv):
        return res.dot(cov_mat_inv).dot(res)

    def _test_consistency(self, constrained_fit, par_cov_mat_inv):
        constrained_fit.do_fit()
        _cost_function = constrained_fit._fitter._fcn_wrapper
        for _i in range(4):
            for _j in range(9):
                _profile_constrained = _cost_function(self._test_par_values[_i, 0, _j], self._test_par_values[_i, 1, _j])
                _diff = _profile_constrained - self._profile_no_constraints[_i, _j]
                _expected_profile_diff = self._expected_profile_diff(self._test_par_res[_i, _j], par_cov_mat_inv)
                self.assertTrue(np.abs(_diff - _expected_profile_diff) < 1e-12)

    @staticmethod
    def _model(a, b):
        return a * np.arange(5) + b

    def setUp(self):
        _data = [-2.1, 0.2, 1.9, 3.8, 6.1]
        self._means = np.array([3.654, 7.789])
        self._vars = np.array([2.467, 1.543])
        self._cov_mat_uncor = np.array([[self._vars[0], 0.0], [0.0, self._vars[1]]])
        self._cov_mat_uncor_inv = np.linalg.inv(self._cov_mat_uncor)
        self._cov_mat_cor = np.array([[self._vars[0], 0.1], [0.1, self._vars[1]]])
        self._cov_mat_cor_inv = np.linalg.inv(self._cov_mat_cor)
        self._cov_mat_simple_a_inv = np.array([[1.0 / self._vars[0], 0.0], [0.0, 0.0]])
        self._cov_mat_simple_b_inv = np.array([[0.0, 0.0], [0.0, 1.0 / self._vars[1]]])

        self._data_container = IndexedContainer(data=_data)
        self._data_container.add_error(err_val=1.0)

        _a_test = np.linspace(start=0,  stop=4, num=9, endpoint=True)
        _b_test = np.linspace(start=-4, stop=0, num=9, endpoint=True)
        self._test_par_values = np.zeros((4, 2, 9))
        self._test_par_values[0, 0] = _a_test
        self._test_par_values[1, 1] = _b_test
        self._test_par_values[2, 0] = _a_test
        self._test_par_values[2, 1] = _b_test
        self._test_par_values[3, 0] = _a_test
        self._test_par_values[3, 1] = -_b_test
        self._test_par_res = self._test_par_values - self._means.reshape((1, 2, 1))
        self._test_par_res = np.transpose(self._test_par_res, axes=(0, 2, 1))

        self._fit_no_constraints = IndexedFit(self._data_container, model_function=self._model)
        self._fit_no_constraints.do_fit()
        _cost_function = self._fit_no_constraints._fitter._fcn_wrapper
        self._profile_no_constraints = np.zeros((4, 9))
        for _i in range(4):
            for _j in range(9):
                self._profile_no_constraints[_i, _j] = _cost_function(
                    self._test_par_values[_i, 0, _j],
                    self._test_par_values[_i, 1, _j])

    def test_bad_input_exception(self):
        _fit_with_constraint = IndexedFit(self._data_container, model_function=self._model)
        with self.assertRaises(IndexedFitException):
            _fit_with_constraint.add_parameter_constraint('c', 1.0, 1.0)
        with self.assertRaises(IndexedFitException):
            _fit_with_constraint.add_matrix_parameter_constraint(['a', 'c'], [1.0, 2.0], [[0.2, 0.0], [0.0, 0.1]])
        with self.assertRaises(IndexedFitException):
            _fit_with_constraint.add_matrix_parameter_constraint(['a'], [1.0, 2.0], [[0.2, 0.0], [0.0, 0.1]])

    def test_fit_profile_cov_mat_uncorrelated(self):
        _fit_with_constraint = IndexedFit(self._data_container, model_function=self._model)
        _fit_with_constraint.add_matrix_parameter_constraint(['a', 'b'], self._means, self._cov_mat_uncor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_uncor_inv)
        _fit_with_constraint_alt = IndexedFit(self._data_container, model_function=self._model)
        _fit_with_constraint_alt.add_parameter_constraint('a', self._means[0], np.sqrt(self._vars[0]))
        _fit_with_constraint_alt.add_parameter_constraint('b', self._means[1], np.sqrt(self._vars[1]))
        self._test_consistency(_fit_with_constraint_alt, self._cov_mat_uncor_inv)

    def test_fit_profile_cov_mat_correlated(self):
        _fit_with_constraint = IndexedFit(self._data_container, model_function=self._model)
        _fit_with_constraint.add_matrix_parameter_constraint(['a', 'b'], self._means, self._cov_mat_cor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_cor_inv)

    def test_fit_profile_simple_a(self):
        _fit_with_constraint = IndexedFit(self._data_container, model_function=self._model)
        _fit_with_constraint.add_parameter_constraint('a', self._means[0], np.sqrt(self._vars[0]))
        self._test_consistency(_fit_with_constraint, self._cov_mat_simple_a_inv)

    def test_fit_profile_simple_b(self):
        _fit_with_constraint = IndexedFit(self._data_container, model_function=self._model)
        _fit_with_constraint.add_parameter_constraint('b', self._means[1], np.sqrt(self._vars[1]))
        self._test_consistency(_fit_with_constraint, self._cov_mat_simple_b_inv)


class TestParameterConstraintInXYFit(unittest.TestCase):

    def _expected_profile_diff(self, res, cov_mat_inv):
        return res.dot(cov_mat_inv).dot(res)

    def _test_consistency(self, constrained_fit, par_cov_mat_inv):
        constrained_fit.do_fit()
        _cost_function = constrained_fit._fitter._fcn_wrapper
        for _i in range(4):
            for _j in range(9):
                _profile_constrained = _cost_function(self._test_par_values[_i, 0, _j], self._test_par_values[_i, 1, _j])
                _diff = _profile_constrained - self._profile_no_constraints[_i, _j]
                _expected_profile_diff = self._expected_profile_diff(self._test_par_res[_i, _j], par_cov_mat_inv)
                self.assertTrue(np.abs(_diff - _expected_profile_diff) < 1e-12)

    def setUp(self):
        _x = [ 0.0, 1.0, 2.0, 3.0, 4.0]
        _y = [-2.1, 0.2, 1.9, 3.8, 6.1]
        self._means = np.array([3.654, 7.789])
        self._vars = np.array([2.467, 1.543])
        self._cov_mat_uncor = np.array([[self._vars[0], 0.0], [0.0, self._vars[1]]])
        self._cov_mat_uncor_inv = np.linalg.inv(self._cov_mat_uncor)
        self._cov_mat_cor = np.array([[self._vars[0], 0.1], [0.1, self._vars[1]]])
        self._cov_mat_cor_inv = np.linalg.inv(self._cov_mat_cor)
        self._cov_mat_simple_a_inv = np.array([[1.0 / self._vars[0], 0.0], [0.0, 0.0]])
        self._cov_mat_simple_b_inv = np.array([[0.0, 0.0], [0.0, 1.0 / self._vars[1]]])

        self._data_container = XYContainer(x_data=_x, y_data=_y)
        self._data_container.add_error(axis='y', err_val=1.0)

        _a_test = np.linspace(start=0,  stop=4, num=9, endpoint=True)
        _b_test = np.linspace(start=-4, stop=0, num=9, endpoint=True)
        self._test_par_values = np.zeros((4, 2, 9))
        self._test_par_values[0, 0] = _a_test
        self._test_par_values[1, 1] = _b_test
        self._test_par_values[2, 0] = _a_test
        self._test_par_values[2, 1] = _b_test
        self._test_par_values[3, 0] = _a_test
        self._test_par_values[3, 1] = -_b_test
        self._test_par_res = self._test_par_values - self._means.reshape((1, 2, 1))
        self._test_par_res = np.transpose(self._test_par_res, axes=(0, 2, 1))

        self._fit_no_constraints = XYFit(self._data_container)
        self._fit_no_constraints.do_fit()
        _cost_function = self._fit_no_constraints._fitter._fcn_wrapper
        self._profile_no_constraints = np.zeros((4, 9))
        for _i in range(4):
            for _j in range(9):
                self._profile_no_constraints[_i, _j] = _cost_function(
                    self._test_par_values[_i, 0, _j],
                    self._test_par_values[_i, 1, _j])

    def test_bad_input_exception(self):
        _fit_with_constraint = XYFit(self._data_container)
        with self.assertRaises(XYFitException):
            _fit_with_constraint.add_parameter_constraint('c', 1.0, 1.0)
        with self.assertRaises(XYFitException):
            _fit_with_constraint.add_matrix_parameter_constraint(['a', 'c'], [1.0, 2.0], [[0.2, 0.0], [0.0, 0.1]])
        with self.assertRaises(XYFitException):
            _fit_with_constraint.add_matrix_parameter_constraint(['a'], [1.0, 2.0], [[0.2, 0.0], [0.0, 0.1]])

    def test_fit_profile_cov_mat_uncorrelated(self):
        _fit_with_constraint = XYFit(self._data_container)
        _fit_with_constraint.add_matrix_parameter_constraint(['a', 'b'], self._means, self._cov_mat_uncor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_uncor_inv)
        _fit_with_constraint_alt = XYFit(self._data_container)
        _fit_with_constraint_alt.add_parameter_constraint('a', self._means[0], np.sqrt(self._vars[0]))
        _fit_with_constraint_alt.add_parameter_constraint('b', self._means[1], np.sqrt(self._vars[1]))
        self._test_consistency(_fit_with_constraint_alt, self._cov_mat_uncor_inv)

    def test_fit_profile_cov_mat_correlated(self):
        _fit_with_constraint = XYFit(self._data_container)
        _fit_with_constraint.add_matrix_parameter_constraint(['a', 'b'], self._means, self._cov_mat_cor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_cor_inv)

    def test_fit_profile_simple_a(self):
        _fit_with_constraint = XYFit(self._data_container)
        _fit_with_constraint.add_parameter_constraint('a', self._means[0], np.sqrt(self._vars[0]))
        self._test_consistency(_fit_with_constraint, self._cov_mat_simple_a_inv)

    def test_fit_profile_simple_b(self):
        _fit_with_constraint = XYFit(self._data_container)
        _fit_with_constraint.add_parameter_constraint('b', self._means[1], np.sqrt(self._vars[1]))
        self._test_consistency(_fit_with_constraint, self._cov_mat_simple_b_inv)
