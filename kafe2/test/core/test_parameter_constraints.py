import unittest2 as unittest
import numpy as np

from kafe2.core.constraint import GaussianMatrixParameterConstraint, GaussianSimpleParameterConstraint


class TestMatrixParameterConstraintDirect(unittest.TestCase):

    def setUp(self):
        self._fit_par_values = [0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0]
        self._par_test_values = np.array([[8.9, 3.4, 5.6], [0.1, 2.3, 9.0], [5.6, 4.5, 3.4]])
        self._par_indices = [[8, 3, 5], [0, 2, 9], [5, 4, 3]]
        self._par_values = np.array([[1.23, 7.20, 3.95], [4.11, 3.00, 2.95], [0.1, -8.5, 67.0]])
        self._par_cov_mats_abs = np.array([
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
        self._par_cov_mats_rel = np.array([
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3],
            ], [
                [0.10, 0.01, 0.02],
                [0.01, 0.20, 0.03],
                [0.02, 0.03, 0.30],
            ]
        ])
        self._uncertainties_abs = np.array([[1.2, 2.3, 0.4], [6.5, 2.6, 1.0]])
        self._uncertainties_rel = np.array([[0.2, 0.3, 0.2], [0.5, 2.6, 1.0]])
        self._par_cor_mats = np.array([
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0]
            ]
        ])

        self._expected_cost_cov_abs = np.zeros((3, 3, 2))
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    _res = self._par_test_values[_i] - self._par_values[_j]
                    self._expected_cost_cov_abs[_i, _j, _k] = _res.dot(np.linalg.inv(self._par_cov_mats_abs[_k])).dot(_res)
        self._expected_cost_cov_rel = np.zeros((3, 3, 2))
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    _res = self._par_test_values[_i] - self._par_values[_j]
                    _abs_cov_mat = self._par_cov_mats_rel[_k] * np.outer(self._par_values[_j], self._par_values[_j])
                    self._expected_cost_cov_rel[_i, _j, _k] = _res.dot(np.linalg.inv(_abs_cov_mat)).dot(_res)
        self._expected_cost_cor_abs = np.zeros((3, 3, 2, 2))
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    for _l in range(2):
                        _res = self._par_test_values[_i] - self._par_values[_j]
                        _cov_mat = self._par_cor_mats[_k] * np.outer(self._uncertainties_abs[_l],
                                                                         self._uncertainties_abs[_l])
                        self._expected_cost_cor_abs[_i, _j, _k, _l] = _res.dot(np.linalg.inv(_cov_mat)).dot(_res)
        self._expected_cost_cor_rel = np.zeros((3, 3, 2, 2))
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    for _l in range(2):
                        _res = self._par_test_values[_i] - self._par_values[_j]
                        _uncertainties_abs = self._uncertainties_rel[_l] * self._par_values[_j]
                        _cov_mat = self._par_cor_mats[_k] * np.outer(_uncertainties_abs, _uncertainties_abs)
                        self._expected_cost_cor_rel[_i, _j, _k, _l] = _res.dot(np.linalg.inv(_cov_mat)).dot(_res)

    def _call_all_properties(self, matrix_constraint):
        matrix_constraint.indices
        matrix_constraint.values
        matrix_constraint.cov_mat
        matrix_constraint.cov_mat_rel
        matrix_constraint.cor_mat
        matrix_constraint.uncertainties
        matrix_constraint.uncertainties_rel
        matrix_constraint.matrix_type
        matrix_constraint.relative
        matrix_constraint.cov_mat_inverse

    def test_bad_input_errors(self):
        with self.assertRaises(ValueError):  # matrix not symmetric
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[0.0, 1.0], matrix=[[0.5, 0.1], [0.0, 0.6]])
        with self.assertRaises(ValueError):  # values wrong dim
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[[0.0, 1.0]], matrix=[[0.5, 0.0], [0.0, 0.6]])
        with self.assertRaises(ValueError):  # values wrong length
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[0.0, 1.0, 5.0], matrix=[[0.5, 0.0], [0.0, 0.6]])
        with self.assertRaises(ValueError):  # both uncertainties and cov mat
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[0.0, 1.0], matrix=[[0.5, 0.0], [0.0, 0.6]],
                                              uncertainties=[0.5, 0.6])
        with self.assertRaises(ValueError):  # unknown matrix type
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[0.0, 1.0], matrix=[[0.5, 0.0], [0.0, 0.6]],
                                              matrix_type='cost')
        with self.assertRaises(ValueError):  # cor_mat but no uncertainties
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[0.0, 1.0], matrix=[[1.0, 0.0], [0.0, 1.0]],
                                              matrix_type='cor')
        with self.assertRaises(ValueError):  # cor mat diagonal elements != 1
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[0.0, 1.0], matrix=[[1.1, 0.0], [0.0, 1.0]],
                                              matrix_type='cor')
        with self.assertRaises(ValueError):  # cor mat elements > 1
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[0.0, 1.0], matrix=[[1.0, 2.0], [2.0, 1.0]],
                                              matrix_type='cor')
        with self.assertRaises(ValueError):  # cor mat elements < -1
            GaussianMatrixParameterConstraint(indices=[1, 0], values=[0.0, 1.0], matrix=[[1.0, -2.0], [-2.0, 1.0]],
                                              matrix_type='cor')

    def test_cost_matrix_cov_abs(self):
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    _constraint = GaussianMatrixParameterConstraint(
                        self._par_indices[_i], self._par_values[_j], self._par_cov_mats_abs[_k])
                    self.assertTrue(np.abs(
                        _constraint.cost(self._fit_par_values) - self._expected_cost_cov_abs[_i, _j, _k]) < 1e-12)
                    self._call_all_properties(_constraint)

    def test_cost_matrix_cov_rel(self):
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    _constraint = GaussianMatrixParameterConstraint(
                        self._par_indices[_i], self._par_values[_j], self._par_cov_mats_rel[_k],
                        relative=True
                    )
                    self.assertTrue(np.abs(
                        _constraint.cost(self._fit_par_values) - self._expected_cost_cov_rel[_i, _j, _k]) < 1e-12)
                    self._call_all_properties(_constraint)

    def test_cost_matrix_cor_abs(self):
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    for _l in range(2):
                        _constraint = GaussianMatrixParameterConstraint(
                            self._par_indices[_i], self._par_values[_j], self._par_cor_mats[_k],
                            matrix_type='cor', uncertainties=self._uncertainties_abs[_l]
                        )
                        self.assertTrue(np.abs(
                            _constraint.cost(self._fit_par_values)
                            - self._expected_cost_cor_abs[_i, _j, _k, _l]) < 1e-12)
                        self._call_all_properties(_constraint)

    def test_cost_matrix_cor_rel(self):
        for _i in range(3):
            for _j in range(3):
                for _k in range(2):
                    for _l in range(2):
                        _constraint = GaussianMatrixParameterConstraint(
                            self._par_indices[_i], self._par_values[_j], self._par_cor_mats[_k],
                            matrix_type='cor', uncertainties=self._uncertainties_rel[_l], relative=True
                        )
                        self.assertTrue(np.abs(
                            _constraint.cost(self._fit_par_values)
                            - self._expected_cost_cor_rel[_i, _j, _k, _l]) < 1e-12)
                        self._call_all_properties(_constraint)


class TestSimpleParameterConstraintDirect(unittest.TestCase):

    def setUp(self):
        self._fit_par_values = [0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0]
        self._par_test_values = [8.9, 3.4, 5.6]
        self._par_indices = [8, 3, 5]
        self._par_values = [1.23, 7.20, 3.95]
        self._par_uncertainties_abs = [1.0, 2.8, 0.001]
        self._par_uncertainties_rel = [0.1, 0.3, 0.01]
        self._expected_cost_abs = np.zeros((3, 3, 3))
        for _i in range(3):
            for _j in range(3):
                for _k in range(3):
                    _res = self._par_test_values[_i] - self._par_values[_j]
                    self._expected_cost_abs[_i, _j, _k] = (_res / self._par_uncertainties_abs[_k]) ** 2
        self._expected_cost_rel = np.zeros((3, 3, 3))
        for _i in range(3):
            for _j in range(3):
                for _k in range(3):
                    _res = self._par_test_values[_i] - self._par_values[_j]
                    self._expected_cost_rel[_i, _j, _k] = (_res / (self._par_uncertainties_rel[_k]
                                                                   * self._par_values[_j])) ** 2

    def test_cost_simple_abs(self):
        for _i in range(3):
            for _j in range(3):
                for _k in range(3):
                    _constraint = GaussianSimpleParameterConstraint(
                        self._par_indices[_i], self._par_values[_j], self._par_uncertainties_abs[_k])
                    self.assertTrue(np.allclose(
                        _constraint.cost(self._fit_par_values), self._expected_cost_abs[_i, _j, _k]))

                    # ensure that results are consistent with matrix constraints
                    _constraint = GaussianMatrixParameterConstraint(
                        [self._par_indices[_i]], [self._par_values[_j]], [[self._par_uncertainties_abs[_k] ** 2]]
                    )
                    self.assertTrue(np.allclose(
                        _constraint.cost(self._fit_par_values), self._expected_cost_abs[_i, _j, _k]))
                    _constraint = GaussianMatrixParameterConstraint(
                        [self._par_indices[_i]], [self._par_values[_j]], [[1.0]],
                        matrix_type='cor', uncertainties=[self._par_uncertainties_abs[_k]]
                    )
                    self.assertTrue(np.allclose(
                        _constraint.cost(self._fit_par_values), self._expected_cost_abs[_i, _j, _k]))

    def test_cost_simple_rel(self):
        for _i in range(3):
            for _j in range(3):
                for _k in range(3):
                    _constraint = GaussianSimpleParameterConstraint(
                        self._par_indices[_i], self._par_values[_j], self._par_uncertainties_rel[_k],
                        relative=True
                    )
                    self.assertTrue(np.allclose(
                        _constraint.cost(self._fit_par_values), self._expected_cost_rel[_i, _j, _k]))

                    # ensure that results are consistent with matrix constraints
                    _constraint = GaussianMatrixParameterConstraint(
                        [self._par_indices[_i]], [self._par_values[_j]], [[self._par_uncertainties_rel[_k] ** 2]],
                        relative=True
                    )
                    self.assertTrue(np.allclose(
                        _constraint.cost(self._fit_par_values), self._expected_cost_rel[_i, _j, _k]))
                    _constraint = GaussianMatrixParameterConstraint(
                        [self._par_indices[_i]], [self._par_values[_j]], [[1.0]],
                        matrix_type='cor', uncertainties=[self._par_uncertainties_rel[_k]], relative=True
                    )
                    self.assertTrue(np.allclose(
                        _constraint.cost(self._fit_par_values), self._expected_cost_rel[_i, _j, _k]))
