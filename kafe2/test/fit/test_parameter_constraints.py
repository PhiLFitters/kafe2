import unittest
import numpy as np

from kafe2.fit import XYContainer, XYFit
from kafe2.fit._base.constraint import GaussianMatrixParameterConstraint, GaussianSimpleParameterConstraint, \
    ParameterConstraintException
from kafe2.fit._base.fit import FitException


class TestMatrixParameterConstraintDirect(unittest.TestCase):

    def setUp(self):
        self._fit_par_values = [0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0]
        self._par_values = np.array([[8.9, 3.4, 5.6], [0.1, 2.3, 9.0], [5.6, 4.5, 3.4]])
        self._par_indices = [[8, 3, 5], [0, 2, 9], [5, 4, 3]]
        self._par_means = np.array([[1.23, 7.20, 3.95], [4.11, 3.00, 2.95], [0.1, -8.5, 67.0]])
        self._par_cov_mats = np.array([
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.8, 0.0],
                [0.0, 0.0, 0.5],
            ], [
                [1.0, 0.2, 0.3],
                [0.2, 2.8, 0.1],
                [0.3, 0.1, 0.5],
            ]
        ])
        self._expected_cost = np.zeros((3, 3, 2))
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    _res = self._par_values[_i] - self._par_means[_j]
                    self._expected_cost[_i, _j, _k] = _res.dot(np.linalg.inv(self._par_cov_mats[_k])).dot(_res)

    def test_bad_input_exception(self):
        with self.assertRaises(ParameterConstraintException):
            GaussianMatrixParameterConstraint([1, 0], [0.0, 1.0], [[0.5, 0.1], [0.0, 0.6]])
        with self.assertRaises(ParameterConstraintException):
            GaussianMatrixParameterConstraint([1, 0], [[0.0, 1.0]], [[0.5, 0.0], [0.0, 0.6]])
        with self.assertRaises(ParameterConstraintException):
            GaussianMatrixParameterConstraint([1, 0], [0.0, 1.0, 5.0], [[0.5, 0.0], [0.0, 0.6]])

    def test_cost_matrix(self):
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    _constraint = GaussianMatrixParameterConstraint(
                        self._par_indices[_i], self._par_means[_j], self._par_cov_mats[_k])
                    self.assertTrue(np.abs(
                        _constraint.cost(self._fit_par_values) - self._expected_cost[_i, _j, _k]) < 1e-12)


class TestSimpleParameterConstraintDirect(unittest.TestCase):

    def setUp(self):
        self._fit_par_values = [0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0]
        self._par_values = [8.9, 3.4, 5.6]
        self._par_indices = [8, 3, 5]
        self._par_means = [1.23, 7.20, 3.95]
        self._par_uncertainties = [1.0, 2.8, 0.001]
        self._expected_cost = np.zeros((3, 3, 3))
        for _i in range(3):
            for _j in range(3):
                for _k in range(3):
                    _res = self._par_values[_i] - self._par_means[_j]
                    self._expected_cost[_i, _j, _k] = (_res / self._par_uncertainties[_k]) ** 2

    def test_cost_simple(self):
        for _i in range(3):
            for _j in range(3):
                for _k in range(3):
                    _constraint = GaussianSimpleParameterConstraint(
                        self._par_indices[_i], self._par_means[_j], self._par_uncertainties[_k])
                    self.assertTrue(np.abs(
                        _constraint.cost(self._fit_par_values) - self._expected_cost[_i, _j, _k]) < 1e-12)


class TestParameterConstraintInXYFit(unittest.TestCase):

    def _expected_profile_diff(self, res, cov_mat_inv):
        return res.dot(cov_mat_inv).dot(res)

    def _test_consistency(self, constrained_fit, par_cov_mat):
        constrained_fit.do_fit()
        _cost_function = constrained_fit._fitter._fcn_wrapper
        for _i in range(4):
            for _j in range(9):
                _profile_constrained = _cost_function(self._test_par_values[_i, 0, _j], self._test_par_values[_i, 1, _j])
                _diff = _profile_constrained - self._profile_no_constraints[_i, _j]
                _expected_profile_diff = self._expected_profile_diff(self._test_par_res[_i, _j], par_cov_mat)
                self.assertTrue(np.abs(_diff - _expected_profile_diff) < 1e-12)

    def setUp(self):
        _x = [ 0.0, 1.0, 2.0, 3.0, 4.0]
        _y = [-2.0, 0.0, 2.0, 4.0, 6.0]
        self._means = np.array([3.654, 7.789])
        self._vars = np.array([2.467, 1.543])
        self._cov_mat_uncor = np.array([[self._vars[0], 0.0], [0.0, self._vars[1]]])
        self._cov_mat_uncor_inv = np.linalg.inv(self._cov_mat_uncor)
        self._cov_mat_cor = np.array([[self._vars[0], 0.1], [0.1, self._vars[1]]])
        self._cov_mat_cor_inv = np.linalg.inv(self._cov_mat_cor)
        self._cov_mat_simple_a_inv = np.array([[1.0 / self._vars[0], 0.0], [0.0, 0.0]])
        self._cov_mat_simple_b_inv = np.array([[0.0, 0.0], [0.0, 1.0 / self._vars[1]]])

        self._data_container = XYContainer(x_data=_x, y_data=_y)
        self._data_container.add_simple_error(axis='y', err_val=1.0)

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
        with self.assertRaises(FitException):
            _fit_with_constraint.add_parameter_constraint('c', 1.0, 1.0)
        with self.assertRaises(FitException):
            _fit_with_constraint.add_matrix_parameter_constraint(['a', 'c'], [1.0, 2.0], [[0.2, 0.0], [0.0, 0.1]])
        with self.assertRaises(FitException):
            _fit_with_constraint.add_matrix_parameter_constraint(['a'], [1.0, 2.0], [[0.2, 0.0], [0.0, 0.1]])

    def test_fit_profile_cov_mat_uncorrelated(self):
        _fit_with_constraint = XYFit(self._data_container)
        _fit_with_constraint.add_matrix_parameter_constraint(['a', 'b'], self._means, self._cov_mat_uncor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_uncor_inv)

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

