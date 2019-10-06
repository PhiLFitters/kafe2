import unittest2 as unittest
from scipy import optimize as opt

import numpy as np

_cannot_import_ROOT = False
try:
    from kafe2.core.minimizers.root_tminuit_minimizer import MinimizerROOTTMinuit
except ImportError:
    _cannot_import_ROOT = True


def fcn_3(x, y, z):
    return (x - 1.23)**2 + 0.5*(y - 4.32)**2 + 3*(z - 9.81)**2 - 5.23

def fcn_1(x):
    return 42.*(x - 42.)**2 + 42.

@unittest.skipIf(_cannot_import_ROOT, "Cannot import ROOT")
class TestMinimizerIMinuit(unittest.TestCase):

    def setUp(self):
        #self.fun = lambda x, y: opt.rosen(np.array([x, y]))
        self.func3 = fcn_3
        self.m3 = MinimizerROOTTMinuit(function_to_minimize=self.func3,
                                  parameter_names=('x', 'y', 'z'),
                                  parameter_values=(1.0, 1.0, 1.0),
                                  parameter_errors=(0.1, 0.1, 0.1))
        self.m3.tolerance = 0.001

        self.func1 = fcn_1
        self.m1 = MinimizerROOTTMinuit(function_to_minimize=self.func1,
                                  parameter_names=('x',),
                                  parameter_values=(1.0,),
                                  parameter_errors=(0.1,))
        self.m1.tolerance = 0.001

        self._ref_hessian_fcn3 = np.diag([1 * 2, 0.5 * 2, 3 * 2])
        self._ref_hessian_inv_fcn3 = np.diag([1./2., 1., 1./6.])

        self._ref_cov_mat_fcn3 = np.diag([1, 1. / 0.5, 1. / 3.])
        self._ref_cor_mat_fcn3 = np.eye(3)

        self._ref_cov_mat_fcn3_fix_x = np.diag([0., 1. / 0.5, 1. / 3.])
        self._ref_cor_mat_fcn3_fix_x = np.diag([0., 1., 1.])

        self._ref_scipy_method = 'slsqp'  # solver algorithm name

        def fcn_3_wrapper(args):
            return fcn_3(*args)

        self._scipy_fmin = opt.minimize(fcn_3_wrapper, (1.,1.,1.), method=self._ref_scipy_method)

    def test_compare_par_values_minimize_fcn1(self):
        self.m1.minimize()
        _v = self.m1.parameter_values
        self.assertAlmostEqual(_v[0], 42., places=4)

    def test_compare_par_errors_minimize_fcn1(self):
        self.m1.minimize()
        _e = self.m1.parameter_errors
        self.assertAlmostEqual(_e[0], np.sqrt(1./42.), places=4)

    def test_compare_func_value_minimize_fcn1(self):
        self.m1.minimize()
        _fv = self.m1.function_value
        self.assertAlmostEqual(_fv, 42., places=4)

    def test_compare_par_values_minimize_fcn3(self):
        self.m3.minimize()
        _v = self.m3.parameter_values
        self.assertAlmostEqual(_v[0], 1.23, places=4)
        self.assertAlmostEqual(_v[1], 4.32, places=4)
        self.assertAlmostEqual(_v[2], 9.81, places=4)

    def test_compare_par_errors_minimize_fcn3(self):
        self.m3.minimize()
        _e = self.m3.parameter_errors
        self.assertAlmostEqual(_e[0], 1., places=4)
        self.assertAlmostEqual(_e[1], np.sqrt(1./0.5), places=4)
        self.assertAlmostEqual(_e[2], np.sqrt(1./3.), places=4)

    def test_compare_func_value_minimize_fcn3(self):
        self.m3.minimize()
        _fv = self.m3.function_value
        self.assertAlmostEqual(_fv, -5.23, places=4)

    def test_compare_par_values_minimize_fcn3_fix_release_x(self):
        self.m3.fix('x')
        self.m3.minimize()
        _v = self.m3.parameter_values
        self.assertAlmostEqual(_v[0], 1., places=4)
        self.assertAlmostEqual(_v[1], 4.32, places=4)
        self.assertAlmostEqual(_v[2], 9.81, places=4)

        self.m3.release('x')
        self.m3.minimize()
        _v = self.m3.parameter_values
        self.assertAlmostEqual(_v[0], 1.23, places=4)
        self.assertAlmostEqual(_v[1], 4.32, places=4)
        self.assertAlmostEqual(_v[2], 9.81, places=4)


    def test_compare_par_values_minimize_fcn3_limit_unlimit_x(self):
        self.m3.limit('x', (-1., 0.7))
        self.m3.minimize()
        _v = self.m3.parameter_values
        self.assertAlmostEqual(_v[0], 0.7, places=4)
        self.assertAlmostEqual(_v[1], 4.32, places=4)
        self.assertAlmostEqual(_v[2], 9.81, places=4)

        self.m3.unlimit('x')
        self.m3.minimize()
        _v = self.m3.parameter_values
        self.assertAlmostEqual(_v[0], 1.23, places=4)
        self.assertAlmostEqual(_v[1], 4.32, places=4)
        self.assertAlmostEqual(_v[2], 9.81, places=4)

    def test_compare_cov_mat_minimize_fcn3(self):
        self.m3.minimize()
        _cm = self.m3.cov_mat
        self.assertTrue(
            np.allclose(_cm, self._ref_cov_mat_fcn3, atol=1e-7)
        )

    def test_compare_cor_mat_minimize_fcn3(self):
        self.m3.minimize()
        _cm = self.m3.cor_mat
        self.assertTrue(
            np.allclose(_cm, self._ref_cor_mat_fcn3, atol=1e-7)
        )

    def test_compare_cov_mat_minimize_fcn3_fix_x(self):
        self.m3.fix('x')
        self.m3.minimize()
        _cm = self.m3.cov_mat
        self.assertTrue(
            np.allclose(_cm, self._ref_cov_mat_fcn3_fix_x, atol=1e-7)
        )

    def test_compare_cor_mat_minimize_fcn3_fix_x(self):
        self.m3.fix('x')
        self.m3.minimize()
        _cm = self.m3.cor_mat
        self.assertTrue(
            np.allclose(_cm, self._ref_cor_mat_fcn3_fix_x, atol=1e-7)
        )

    def test_compare_par_values_minimize_fcn1_errdef_4(self):
        self.m1.errordef = 4.0
        self.m1.minimize()
        _v = self.m1.parameter_values
        self.assertAlmostEqual(_v[0], 42., places=4)

    def test_compare_par_errors_minimize_fcn1_errdef_4(self):
        self.m1.errordef = 4.0
        self.m1.minimize()
        _e = self.m1.parameter_errors
        self.assertAlmostEqual(_e[0], np.sqrt(4.0 *(1. / 42.)), places=4)

    def test_compare_cov_mat_minimize_fcn3_errdef_4(self):
        self.m3.errordef = 4.0
        self.m3.minimize()
        _cm = self.m3.cov_mat
        self.assertTrue(
            np.allclose(_cm, 4.0 * self._ref_cov_mat_fcn3, atol=1e-7)
        )

    def test_compare_hessian_minimize_fcn3(self):
        #self.m3.errordef = 4.0
        self.m3.minimize()
        _hm = self.m3.hessian
        self.assertTrue(
            np.allclose(_hm, self._ref_hessian_fcn3, atol=1e-7)
        )

    def test_compare_hessian_minimize_fcn3_errdef_4(self):
        self.m3.errordef = 4.0
        self.m3.minimize()
        _hm = self.m3.hessian
        self.assertTrue(
            np.allclose(_hm, self._ref_hessian_fcn3, atol=1e-7)
        )

    def test_compare_hessian_inv_minimize_fcn3(self):
        #self.m3.errordef = 4.0
        self.m3.minimize()
        _hm = self.m3.hessian_inv
        self.assertTrue(
            np.allclose(_hm, self._ref_hessian_inv_fcn3, atol=1e-7)
        )

    def test_compare_hessian_inv_minimize_fcn3_errdef_4(self):
        self.m3.errordef = 4.0
        self.m3.minimize()
        _hm = self.m3.hessian_inv
        self.assertTrue(
            np.allclose(_hm, self._ref_hessian_inv_fcn3, atol=1e-7)
        )

    def test_compare_par_values_to_scipy_optimize(self):
        self.m3.minimize()
        _vs = self.m3.parameter_values
        for _v_scipy, _v in zip(self._scipy_fmin.x, _vs):
            self.assertAlmostEqual(_v_scipy, _v, places=2)

    @unittest.skip("SLSQP does not compute hessian by default")
    def test_compare_hessian_inv_to_scipy_optimize(self):
        self.m3.minimize()
        _hm_inv = self.m3.hessian_inv
        _hm_inv_scipy = self._scipy_fmin.hess_inv
        self.assertTrue(
            np.allclose(_hm_inv, _hm_inv_scipy, atol=1e-2)
        )
