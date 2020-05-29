import unittest2 as unittest

import numpy as np

from kafe2.core.error import CovMat, cov_mat_from_float_list, cov_mat_from_float


class TestCovMat(unittest.TestCase):

    def setUp(self):
        cm = CovMat([
            [1.00000000e-02, 1.80600000e-02, 8.19000000e-03, 5.62800000e-02, 1.26000000e-03],
            [1.80600000e-02, 1.84900000e-01, 3.52170000e-02, 2.42004000e-01, 5.41800000e-03],
            [8.19000000e-03, 3.52170000e-02, 3.80250000e-02, 1.09746000e-01, 2.45700000e-03],
            [5.62800000e-02, 2.42004000e-01, 1.09746000e-01, 1.79560000e+00, 1.68840000e-02],
            [1.26000000e-03, 5.41800000e-03, 2.45700000e-03, 1.68840000e-02, 9.00000000e-04]
        ])
        self.reference = [2., 1., 3., 2., 9.]
        self.cm = cm
        self._cm_add_1 = CovMat([[1.0, 0.0], [0.0, 1.0]])
        self._cm_add_2 = CovMat([[2.0, 1.0], [1.0, 2.0]])
        self._cm_add_3 = CovMat([[3.0, 1.0], [1.0, 3.0]])
        self._cm_chol_fail = CovMat([[0.0, 1.0], [1.0, 0.0]])

    def test_magic(self):
        self.assertEqual(self._cm_add_1 + self._cm_add_2, self._cm_add_3)
        self.assertEqual(self._cm_add_1 + self._cm_add_2, self._cm_add_3.mat)
        self._cm_add_1 += self._cm_add_2
        self.assertEqual(self._cm_add_1, self._cm_add_3)
        self.assertEqual(self._cm_add_1, self._cm_add_3.mat)
        self.assertEqual(self._cm_add_3, self._cm_add_1.mat)
        self.assertTrue(np.all(self._cm_add_1.mat == self._cm_add_3.mat))
        self.assertEqual(len(self.cm), 5)
        self.assertEqual(len(self._cm_add_1), 2)
        self.assertEqual(len(self._cm_add_2), 2)
        self.assertEqual(len(self._cm_add_3), 2)

    def test_not_square_raise(self):
        with self.assertRaises(ValueError):
            CovMat(np.arange(10))

    def test_rescale(self):
        self.cm.rescale(self.reference, [3, 2, 1, 9, 2])
        _ref = np.array([
            [2.25000000e-02, 5.41800000e-02, 4.09500000e-03, 3.79890000e-01, 4.20000000e-04],
            [5.41800000e-02, 7.39600000e-01, 2.34780000e-02, 2.17803600e+00, 2.40800000e-03],
            [4.09500000e-03, 2.34780000e-02, 4.22500000e-03, 1.64619000e-01, 1.82000000e-04],
            [3.79890000e-01, 2.17803600e+00, 1.64619000e-01, 3.63609000e+01, 1.68840000e-02],
            [4.20000000e-04, 2.40800000e-03, 1.82000000e-04, 1.68840000e-02, 4.44444444e-05]
        ])
        self.assertTrue(np.allclose(self.cm.mat, _ref))

    def test_inverse(self):
        self.assertTrue(np.allclose(self.cm.I.dot(self.cm.mat), np.eye(self.cm.mat.shape[0])))

    def test_chol_fail(self):
        self.assertIs(self._cm_chol_fail.chol, None)

    def test_cond(self):
        self.assertEqual(CovMat([[1.0, 0.0], [0.0, 10.0]]).cond, 10.0)

    def test_split_svd(self):
        self.assertTrue(np.allclose(self.cm.mat, np.sum(self.cm.split_svd, axis=0)))
        self.assertIs(self._cm_chol_fail.split_svd, None)


class TestCovMatHelperFunctions(unittest.TestCase):

    def setUp(self):
        self.values = [0.1, 0.43, 0.195, 1.34, 0.03]

        self.ref_fullcor = np.array([
            [1.00000000e-02, 4.30000000e-02, 1.95000000e-02, 1.34000000e-01, 3.00000000e-03],
            [4.30000000e-02, 1.84900000e-01, 8.38500000e-02, 5.76200000e-01, 1.29000000e-02],
            [1.95000000e-02, 8.38500000e-02, 3.80250000e-02, 2.61300000e-01, 5.85000000e-03],
            [1.34000000e-01, 5.76200000e-01, 2.61300000e-01, 1.79560000e+00, 4.02000000e-02],
            [3.00000000e-03, 1.29000000e-02, 5.85000000e-03, 4.02000000e-02, 9.00000000e-04]
        ])

        self.ref_nocor = np.array([
            [1.00000000e-02,            0.0,            0.0,            0.0,            0.0],
            [           0.0, 1.84900000e-01,            0.0,            0.0,            0.0],
            [           0.0,            0.0, 3.80250000e-02,            0.0,            0.0],
            [           0.0,            0.0,            0.0, 1.79560000e+00,            0.0],
            [           0.0,            0.0,            0.0,            0.0, 9.00000000e-04]
        ])

        self.ref_partcor = np.array([
            [1.00000000e-02, 1.80600000e-02, 8.19000000e-03, 5.62800000e-02, 1.26000000e-03],
            [1.80600000e-02, 1.84900000e-01, 3.52170000e-02, 2.42004000e-01, 5.41800000e-03],
            [8.19000000e-03, 3.52170000e-02, 3.80250000e-02, 1.09746000e-01, 2.45700000e-03],
            [5.62800000e-02, 2.42004000e-01, 1.09746000e-01, 1.79560000e+00, 1.68840000e-02],
            [1.26000000e-03, 5.41800000e-03, 2.45700000e-03, 1.68840000e-02, 9.00000000e-04]
        ])

        self.single_value = 0.123

        self.ref_singleval_fullcor = np.array([
            [0.015129, 0.015129, 0.015129],
            [0.015129, 0.015129, 0.015129],
            [0.015129, 0.015129, 0.015129]
        ])

        self.ref_singleval_nocor = np.array([
            [0.015129, 0.      , 0.      ],
            [0.      , 0.015129, 0.      ],
            [0.      , 0.      , 0.015129]
        ])

        self.ref_singleval_partcor = np.array([
            [0.01512900, 0.00635418, 0.00635418],
            [0.00635418, 0.01512900, 0.00635418],
            [0.00635418, 0.00635418, 0.01512900]
        ])

    def test_cov_mat_from_float_list_nocor(self):
        cm = cov_mat_from_float_list(self.values, correlation=0.0)
        self.assertTrue(np.allclose(cm.mat, self.ref_nocor))

    def test_cov_mat_from_float_list_partialcor(self):
        cm = cov_mat_from_float_list(self.values, correlation=0.42)
        self.assertTrue(np.allclose(cm.mat, self.ref_partcor))

    def test_cov_mat_from_float_list_fullcor(self):
        cm = cov_mat_from_float_list(self.values, correlation=1.0)
        self.assertTrue(np.allclose(cm.mat, self.ref_fullcor))

    def test_cov_mat_from_float_list_raise(self):
        with self.assertRaises(ValueError):
            cov_mat_from_float_list(self.values, correlation=2)
        with self.assertRaises(ValueError):
            cov_mat_from_float_list(self.values, correlation=-1)

    def test_cov_mat_from_float_nocor(self):
        cm = cov_mat_from_float(self.single_value, 3, correlation=0.0)
        self.assertTrue(np.allclose(cm.mat, self.ref_singleval_nocor))

    def test_cov_mat_from_float_partcor(self):
        cm = cov_mat_from_float(self.single_value, 3, correlation=0.42)
        self.assertTrue(np.allclose(cm.mat, self.ref_singleval_partcor))

    def test_cov_mat_from_float_fullcor(self):
        cm = cov_mat_from_float(self.single_value, 3, correlation=1.0)
        self.assertTrue(np.allclose(cm.mat, self.ref_singleval_fullcor))

    def test_cov_mat_from_float_raise(self):
        with self.assertRaises(ValueError):
            cov_mat_from_float(self.single_value, 3, correlation=2)
        with self.assertRaises(ValueError):
            cov_mat_from_float(self.single_value, 3, correlation=-1)
