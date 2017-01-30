import unittest
import numpy as np

from kafe.error import SimpleGaussianError, MatrixGaussianError


class TestMatrixGaussianError(unittest.TestCase):

    def setUp(self):
        self.ref_cov_mat = np.matrix([[0.10, 0.04, 0.29],
                                      [0.04, 0.02, 0.42],
                                      [0.29, 0.42, 0.33]])
        self.ref_cor_mat = np.matrix([[ 1.        ,  0.89442719,  1.59639746],
                                      [ 0.89442719,  1.        ,  5.16984262],
                                      [ 1.59639746,  5.16984262,  1.        ]])
        self.ref_err_val = np.sqrt(np.array([0.1, 0.02, 0.33]))

        # -- errors with no reference

        # construct error from cov mat with no reference (absolute error)
        self.ge_cov_noref = MatrixGaussianError(self.ref_cov_mat, 'cov')
        # construct error from cor mat and error array, with no reference (absolute error)
        self.ge_cor_noref = MatrixGaussianError(self.ref_cor_mat, 'cor', err_val=self.ref_err_val)

        # -- errors with reference
        self.ref_reference = [0.1, 0.2, 0.3]
        self.ref_cov_mat_rel = np.matrix([[10.0,   2.0, 9.667],
                                          [ 2.0,   0.5, 7.0  ],
                                          [ 9.667, 7.0, 3.667]])
        self.ref_err_val_rel = np.array([3.16227766, 0.70710678, 1.91494125])

        # construct error from cov mat with reference (absolute error)
        self.ge_cov_wref = MatrixGaussianError(self.ref_cov_mat, 'cov', reference=self.ref_reference)
        # construct error from cor mat and error array, with reference (absolute error)
        self.ge_cor_wref = MatrixGaussianError(self.ref_cor_mat, 'cor', err_val=self.ref_err_val, reference=self.ref_reference)

        # -- relative errors

        # construct error from cov mat with no reference (relative error)
        self.ge_cov_rel_noref = MatrixGaussianError(self.ref_cov_mat_rel, 'cov', relative=True)
        # construct error from cor mat and error array, with no reference (relative error)
        self.ge_cor_rel_noref = MatrixGaussianError(self.ref_cor_mat, 'cor', err_val=self.ref_err_val_rel, relative=True)
        # construct error from cov mat with reference (relative error)
        self.ge_cov_rel_wref = MatrixGaussianError(self.ref_cov_mat_rel, 'cov', relative=True, reference=self.ref_reference)
        # construct error from cor mat and error array, with reference (relative error)
        self.ge_cor_rel_wref = MatrixGaussianError(self.ref_cor_mat, 'cor', err_val=self.ref_err_val_rel, relative=True, reference=self.ref_reference)

        # -- ill defined matrices
        self.wrong_cor_mat_diagonal_not_ones = np.matrix([[42., 0.89442719, 1.59639746],
                                                          [0.89442719, 1., 5.16984262],
                                                          [1.59639746, 5.16984262, 1.]])

        self.wrong_cor_mat_asymm = np.matrix([[1., 42.89442719, 1.59639746],
                                              [0.89442719, 1., 5.16984262],
                                              [1.59639746, 5.16984262, 1.]])


        self.wrong_cov_mat_asymm = np.matrix([[0.10, 42.0, 0.29],
                                              [0.04, 0.02, 0.42],
                                              [0.29, 0.42, 0.33]])

        self.wrong_cov_mat_err_val_mismatch = np.matrix([[0.42, 0.04, 0.29],
                                                         [0.04, 0.42, 0.42],
                                                         [0.29, 0.42, 0.42]])

    def test_compare_cov_from_cov_wref(self):
        self.assertTrue(np.allclose(self.ge_cov_wref.cov_mat, self.ref_cov_mat, atol=1e-3))

    def test_compare_cov_rel_from_cov_rel_wref(self):
        self.assertTrue(np.allclose(self.ge_cov_rel_wref.cov_mat_rel, self.ref_cov_mat_rel, atol=1e-3))

    def test_compare_cov_from_cor_wref(self):
        self.assertTrue(np.allclose(self.ge_cor_wref.cov_mat, self.ref_cov_mat, atol=1e-3))

    def test_compare_cov_rel_from_cor_rel_wref(self):
        self.assertTrue(np.allclose(self.ge_cor_rel_wref.cov_mat_rel, self.ref_cov_mat_rel, atol=1e-3))

    def test_compare_error_from_cov_wref(self):
        self.assertTrue(np.allclose(self.ge_cov_wref.error, self.ref_err_val, atol=1e-3))

    def test_compare_error_rel_from_cov_rel_wref(self):
        self.assertTrue(np.allclose(self.ge_cov_rel_wref.error_rel, self.ref_err_val_rel, atol=1e-3))

    def test_compare_error_from_cor_wref(self):
        self.assertTrue(np.allclose(self.ge_cor_wref.error, self.ref_err_val, atol=1e-3))

    def test_compare_error_rel_from_cor_rel_wref(self):
        self.assertTrue(np.allclose(self.ge_cor_rel_wref.error_rel, self.ref_err_val_rel, atol=1e-3))

    def test_compare_cov_from_cov_noref(self):
        self.assertTrue(np.allclose(self.ge_cov_noref.cov_mat, self.ref_cov_mat, atol=1e-3))

    def test_compare_cov_rel_from_cov_rel_noref(self):
        self.assertTrue(np.allclose(self.ge_cov_rel_noref.cov_mat_rel, self.ref_cov_mat_rel, atol=1e-3))

    def test_compare_cov_from_cor_noref(self):
        self.assertTrue(np.allclose(self.ge_cor_noref.cov_mat, self.ref_cov_mat, atol=1e-3))

    def test_compare_cov_rel_from_cor_rel_noref(self):
        self.assertTrue(np.allclose(self.ge_cor_rel_noref.cov_mat_rel, self.ref_cov_mat_rel, atol=1e-3))

    def test_compare_error_from_cov_noref(self):
        self.assertTrue(np.allclose(self.ge_cov_noref.error, self.ref_err_val, atol=1e-3))

    def test_compare_error_rel_from_cov_rel_noref(self):
        self.assertTrue(np.allclose(self.ge_cov_rel_noref.error_rel, self.ref_err_val_rel, atol=1e-3))

    def test_compare_error_from_cor_noref(self):
        self.assertTrue(np.allclose(self.ge_cor_noref.error, self.ref_err_val, atol=1e-3))

    def test_compare_error_rel_from_cor_rel_noref(self):
        self.assertTrue(np.allclose(self.ge_cor_rel_noref.error_rel, self.ref_err_val_rel, atol=1e-3))

    def test_convert_cov_from_cov_rel_wref(self):
        self.assertTrue(np.allclose(self.ge_cov_rel_wref.cov_mat, self.ref_cov_mat, atol=1e-3))

    def test_convert_cov_rel_from_cov_wref(self):
        self.assertTrue(np.allclose(self.ge_cov_wref.cov_mat_rel, self.ref_cov_mat_rel, atol=1e-3))

    def test_convert_cov_from_cor_rel_wref(self):
        self.assertTrue(np.allclose(self.ge_cor_rel_wref.cov_mat, self.ref_cov_mat, atol=1e-3))

    def test_convert_cov_rel_from_cor_wref(self):
        self.assertTrue(np.allclose(self.ge_cor_wref.cov_mat_rel, self.ref_cov_mat_rel, atol=1e-3))

    def test_convert_error_from_cov_rel_wref(self):
        self.assertTrue(np.allclose(self.ge_cov_rel_wref.error, self.ref_err_val, atol=1e-3))

    def test_convert_error_rel_from_cov_wref(self):
        self.assertTrue(np.allclose(self.ge_cov_wref.error_rel, self.ref_err_val_rel, atol=1e-3))

    def test_convert_error_from_cor_rel_wref(self):
        self.assertTrue(np.allclose(self.ge_cor_rel_wref.error, self.ref_err_val, atol=1e-3))

    def test_convert_error_rel_from_cor_wref(self):
        self.assertTrue(np.allclose(self.ge_cor_wref.error_rel, self.ref_err_val_rel, atol=1e-3))

    def test_raise_cov_from_cov_rel_noref(self):
        with self.assertRaises(AttributeError):
            self.ge_cov_rel_noref.cov_mat

    def test_raise_cov_rel_from_cov_noref(self):
        with self.assertRaises(AttributeError):
            self.ge_cov_noref.cov_mat_rel

    def test_raise_cov_from_cor_rel_noref(self):
        with self.assertRaises(AttributeError):
            self.ge_cor_rel_noref.cov_mat

    def test_raise_cov_rel_from_cor_noref(self):
        with self.assertRaises(AttributeError):
            self.ge_cor_noref.cov_mat_rel

    def test_raise_error_from_cov_rel_noref(self):
        with self.assertRaises(AttributeError):
            self.ge_cov_rel_noref.error

    def test_raise_error_rel_from_cov_noref(self):
        with self.assertRaises(AttributeError):
            self.ge_cov_noref.error_rel

    def test_raise_error_from_cor_rel_noref(self):
        with self.assertRaises(AttributeError):
            self.ge_cor_rel_noref.error

    def test_raise_error_rel_from_cor_noref(self):
        with self.assertRaises(AttributeError):
            self.ge_cor_noref.error_rel

    def test_raise_build_from_wrong_cor_mat_diagonal_not_ones(self):
        with self.assertRaises(ValueError):
            MatrixGaussianError(self.wrong_cor_mat_diagonal_not_ones, 'cor', err_val=self.ref_err_val, reference=self.ref_reference)

    def test_raise_build_from_wrong_cov_mat_asymm(self):
        with self.assertRaises(ValueError):
            MatrixGaussianError(self.wrong_cov_mat_asymm, 'cov', err_val=self.ref_err_val, reference=self.ref_reference)

    def test_raise_build_from_wrong_cor_mat_asymm(self):
        with self.assertRaises(ValueError):
            MatrixGaussianError(self.wrong_cor_mat_asymm, 'cor', err_val=self.ref_err_val, reference=self.ref_reference)

    def test_raise_build_from_wrong_cov_mat_err_val_mismatch(self):
        with self.assertRaises(ValueError):
            MatrixGaussianError(self.wrong_cov_mat_err_val_mismatch, 'cov', err_val=self.ref_err_val, reference=self.ref_reference)



