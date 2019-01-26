import unittest
import numpy as np

from kafe2.fit import XYContainer, XYFit
from kafe2.fit._base.constraint import GaussianParameterConstraint


class TestParameterConstraintDirect(unittest.TestCase):

    def setUp(self):
        self._fit_par_values = [0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0]
        self._par_values = np.array([0.1, 1.2, 8.9, 4.5, 5.6])
        self._par_indices = [0, 1, 8, 4, 5]
        self._par_means = np.array([0.1, 0.0, 8.9, 4.0, 6.1])
        self._par_variances = np.array([1.0, 1.44, 1.0, 2.0, 0.5])
        self._par_cov_mat = np.eye(5) * self._par_variances
        self._expected_cost = 0.0 + 1.2 ** 2 / 1.44 + 0.0 + 0.5 ** 2 / 2.0 + 0.5 ** 2 / 0.5

    def test_calculate_cost_with_cov_mat(self):
        _constraint = GaussianParameterConstraint(self._par_indices, self._par_means, self._par_cov_mat)
        _cost = _constraint.cost(self._fit_par_values)
        self.assertTrue(np.sum(_cost - self._expected_cost) < 1e-9)


class TestParameterConstraintInXYFit(unittest.TestCase):

    def _expected_profile_diff(self, res, cov_mat):
        return res.dot(cov_mat).dot(res)

    def _test_consistency(self, constrained_fit, par_cov_mat):
        constrained_fit.do_fit()
        _cost_function = constrained_fit._fitter._fcn_wrapper
        for _i in range(4):
            for _j in range(9):
                _profile_constrained = _cost_function(self._test_par_values[_i, 0, _j], self._test_par_values[_i, 1, _j])
                _diff = _profile_constrained - self._profile_no_constraints[_i, _j]
                _expected_profile_diff = self._expected_profile_diff(self._test_par_res[_i, _j], par_cov_mat)
                self.assertTrue(_diff - _expected_profile_diff < 1e-9)

    def setUp(self):
        _x = [ 0.0, 1.0, 2.0, 3.0, 4.0]
        _y = [-2.0, 0.0, 2.0, 4.0, 6.0]
        self._means = np.array([3.654, 7.789])
        self._vars = np.array([2.467, 1.543])
        self._cov_mat_uncor = np.array([[self._vars[0], 0.0], [0.0, self._vars[1]]])
        self._cov_mat_uncor_inv = np.linalg.inv(self._cov_mat_uncor)
        self._cov_mat_cor = np.array([[self._vars[0], 0.1], [0.1, self._vars[1]]])
        self._cov_mat_cor_inv = np.linalg.inv(self._cov_mat_cor)
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

        self._fit_no_constraints = XYFit(self._data_container, minimizer='scipy')
        self._fit_no_constraints.do_fit()
        _cost_function = self._fit_no_constraints._fitter._fcn_wrapper
        self._profile_no_constraints = np.zeros((4, 9))
        for _i in range(4):
            for _j in range(9):
                self._profile_no_constraints[_i, _j] = _cost_function(
                    self._test_par_values[_i, 0, _j],
                    self._test_par_values[_i, 1, _j])

    def test_fit_profile_cov_mat_uncorrelated(self):
        _fit_with_constraint = XYFit(self._data_container, minimizer='scipy')
        _fit_with_constraint.add_matrix_constraint(['a', 'b'], self._means, self._cov_mat_uncor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_uncor)

    def test_fit_profile_cov_mat_correlated(self):
        _fit_with_constraint = XYFit(self._data_container, minimizer='scipy')
        _fit_with_constraint.add_matrix_constraint(['a', 'b'], self._means, self._cov_mat_cor)
        self._test_consistency(_fit_with_constraint, self._cov_mat_cor)
