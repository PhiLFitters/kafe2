import six
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.optimize import minimize
import unittest2 as unittest
from kafe2.core.minimizers.minimizer_base import MinimizerException


def fcn_1(x):
    return 42.*(x - 42.)**2 + 42.


def fcn_3(x, y, z):
    return (x - 1.23)**2 + 0.5*(y - 4.32)**2 + 3*(z - 9.81)**2 - 5.23


def fcn_3_wrapper(args):
    return fcn_3(*args)


@six.add_metaclass(ABCMeta)
class TestMinimizerMixin:

    @abstractmethod
    def _get_minimizer(self, parameter_names, parameter_values, parameter_errors,
                       function_to_minimize):
        pass

    @property
    @abstractmethod
    def _expected_tolerance(self):
        pass

    # noinspection PyAttributeOutsideInit
    def setUp(self):
        self.func1 = fcn_1
        self.par_names_fcn1 = ['x']
        self.initial_pars_fcn1 = (1.0,)
        self.initial_errs_fcn1 = (0.1,)
        self.m1 = self._get_minimizer(
            parameter_names=self.par_names_fcn1,
            parameter_values=self.initial_pars_fcn1,
            parameter_errors=self.initial_errs_fcn1,
            function_to_minimize=self.func1,
        )

        self.func3 = fcn_3
        self.par_names_fcn3 = ['x', 'y', 'z']
        self.initial_pars_fcn3 = (1.0, 2.0, 3.0)
        self.initial_errs_fcn3 = (0.1, 0.2, 0.5)
        self.m3 = self._get_minimizer(
            parameter_names=self.par_names_fcn3,
            parameter_values=self.initial_pars_fcn3,
            parameter_errors=self.initial_errs_fcn3,
            function_to_minimize=self.func3,
        )

        self._ref_fval_fcn3 = -5.23
        self._ref_par_val_fcn3 = np.array([1.23, 4.32, 9.81])
        self._ref_par_val_fcn3_fix_x = np.array([1.0, 4.32, 9.81])
        self._ref_par_val_fcn3_limit_x = np.array([0.7, 4.32, 9.81])
        self._ref_par_val_fcn3_limit_x_2 = np.array([3.5, 4.32, 9.81])
        self._ref_par_val_fcn3_fix_x_limit_y = np.array([1.0, 0.5, 9.81])
        self._ref_par_err_fcn3 = 1.0 / np.sqrt(np.array([1.0, 0.5, 3.0]))
        self._ref_par_err_fcn3_fix_x = np.array(self._ref_par_err_fcn3)
        self._ref_par_err_fcn3_fix_x[0] = 0
        self._ref_hessian_fcn3 = np.diag(2.0 * np.array([1.0, 0.5, 3.0]))
        self._ref_hessian_inv_fcn3 = np.diag(1.0 / (2.0 * np.array([1.0, 0.5, 3.0])))
        self._ref_cov_mat_fcn3 = np.diag(1.0 / np.array([1.0, 0.5, 3.0]))
        self._ref_cor_mat_fcn3 = np.eye(3)
        self._ref_hessian_fcn3_fix_x = np.array(self._ref_hessian_fcn3)
        self._ref_hessian_fcn3_fix_x[0, 0] = 0
        self._ref_hessian_inv_fcn3_fix_x = np.array(self._ref_hessian_inv_fcn3)
        self._ref_hessian_inv_fcn3_fix_x[0, 0] = 0
        self._ref_cov_mat_fcn3_fix_x = np.array(self._ref_cov_mat_fcn3)
        self._ref_cov_mat_fcn3_fix_x[0, 0] = 0
        self._ref_cor_mat_fcn3_fix_x = np.array(self._ref_cor_mat_fcn3)
        self._ref_cor_mat_fcn3_fix_x[0, 0] = 0

        self._scipy_fmin = minimize(fcn_3_wrapper, np.ones(3), method="bfgs")

        self._ref_contour_m3_x_y_5 = np.array([
            [0.23, 1.23000000, 2.23, 1.23000000, 0.45822021],
            [4.32, 2.90578644, 4.32, 5.73421356, 5.21928411]
        ])
        self._ref_profile_m3_x_5 = np.array([
            [-0.77, 0.23, 1.23, 2.23, 3.23],
            [ 4.00, 1.00, 0.00, 1.00, 4.00]
        ])
        self._ref_profile_m3_x_5_no_subtract_min = np.array(self._ref_profile_m3_x_5)
        self._ref_profile_m3_x_5_no_subtract_min[1] += self._ref_fval_fcn3

    def test_initial_properties_correct(self):
        for _minimizer, _par_names, _initial_pars, _initial_errs, _func in zip(
                [self.m1, self.m3],
                [self.par_names_fcn1, self.par_names_fcn3],
                [self.initial_pars_fcn1, self.initial_pars_fcn3],
                [self.initial_errs_fcn1, self.initial_errs_fcn3],
                [fcn_1, fcn_3]):
            _num_pars = len(_initial_pars)
            self.assertFalse(_minimizer.did_fit)
            self.assertEqual(_minimizer.errordef, 1.0)
            self.assertIs(type(_minimizer.errordef), float)
            self.assertIs(_minimizer.function_to_minimize, _func)
            self.assertEqual(_minimizer.function_value, _func(*_initial_pars))
            self.assertTrue(type(_minimizer.function_value) in [float, np.float64])
            self.assertEqual(_minimizer.num_pars, _num_pars)
            self.assertIs(type(_minimizer.num_pars), int)
            self.assertTrue(np.all(_minimizer.parameter_values == _initial_pars))
            self.assertIs(type(_minimizer.parameter_values), np.ndarray)
            self.assertTrue(np.all(_minimizer.parameter_errors == _initial_errs))
            self.assertIs(type(_minimizer.parameter_errors), np.ndarray)
            self.assertIs(_minimizer.asymmetric_parameter_errors_if_calculated, None)
            self.assertIs(_minimizer.asymmetric_parameter_errors, None)
            self.assertIs(_minimizer.asymmetric_parameter_errors_if_calculated, None)
            self.assertEqual(_minimizer.parameter_names, _par_names)
            self.assertIs(type(_minimizer.parameter_names), list)
            self.assertEqual(_minimizer.tolerance, self._expected_tolerance)
            self.assertIs(type(_minimizer.tolerance), float)
            self.assertIs(_minimizer.hessian, None)
            self.assertIs(_minimizer.hessian_inv, None)
            self.assertIs(_minimizer.cov_mat, None)
            self.assertIs(_minimizer.cor_mat, None)

    def test_properties_copied(self):
        for _minimizer in [self.m1, self.m3]:
            _minimizer.minimize()
            self.assertIsNot(_minimizer.parameter_values, _minimizer.parameter_values)
            self.assertIsNot(_minimizer.parameter_errors, _minimizer.parameter_errors)
            self.assertIsNot(
                _minimizer.asymmetric_parameter_errors, _minimizer.asymmetric_parameter_errors)
            self.assertIsNot(_minimizer.parameter_names, _minimizer.parameter_names)
            self.assertIsNot(_minimizer.hessian, _minimizer.hessian)
            self.assertIsNot(_minimizer.hessian_inv, _minimizer.hessian_inv)
            self.assertIsNot(_minimizer.cov_mat, _minimizer.cov_mat)
            self.assertIsNot(_minimizer.cor_mat, _minimizer.cor_mat)

    def test_neg_par_errs_raise(self):
        for _minimizer in [self.m1, self.m3]:
            with self.assertRaises(ValueError):
                _minimizer.parameter_errors = -_minimizer.parameter_errors

    def test_did_fit_errordef(self):
        for _minimizer in [self.m1, self.m3]:
            self.assertFalse(_minimizer.did_fit)
            _minimizer.minimize()
            self.assertTrue(_minimizer.did_fit)
            _minimizer.errordef = _minimizer.errordef
            self.assertFalse(_minimizer.did_fit)

    def test_did_fit_parameter_values(self):
        for _minimizer in [self.m1, self.m3]:
            self.assertFalse(_minimizer.did_fit)
            _minimizer.minimize()
            self.assertTrue(_minimizer.did_fit)
            _minimizer.parameter_values = _minimizer.parameter_values
            self.assertFalse(_minimizer.did_fit)

    def test_did_fit_parameter_errors(self):
        for _minimizer in [self.m1, self.m3]:
            self.assertFalse(_minimizer.did_fit)
            _minimizer.minimize()
            self.assertTrue(_minimizer.did_fit)
            _minimizer.parameter_errors = _minimizer.parameter_errors
            self.assertFalse(_minimizer.did_fit)

    def test_did_fit_set(self):
        for _minimizer in [self.m1, self.m3]:
            self.assertFalse(_minimizer.did_fit)
            _minimizer.minimize()
            self.assertTrue(_minimizer.did_fit)
            _minimizer.set(_minimizer.parameter_names[0], _minimizer.parameter_values[0])
            self.assertFalse(_minimizer.did_fit)

    def test_did_fit_tolerance(self):
        for _minimizer in [self.m1, self.m3]:
            self.assertFalse(_minimizer.did_fit)
            _minimizer.minimize()
            self.assertTrue(_minimizer.did_fit)
            _minimizer.tolerance = _minimizer.tolerance
            self.assertFalse(_minimizer.did_fit)

    def test_compare_par_values_minimize_fcn1(self):
        self.m1.minimize()
        _par_vals = self.m1.parameter_values
        self.assertIs(type(_par_vals), np.ndarray)
        self.assertAlmostEqual(_par_vals[0], 42.)

    def test_compare_par_errors_minimize_fcn1(self):
        self.m1.minimize()
        _par_errs = self.m1.parameter_errors
        self.assertIs(type(_par_errs), np.ndarray)
        self.assertAlmostEqual(_par_errs[0], np.sqrt(1.0 / 42.0))

    def test_compare_func_value_minimize_fcn1(self):
        self.m1.minimize()
        _fval = self.m1.function_value
        self.assertTrue(type(_fval) in [float, np.float64])
        self.assertAlmostEqual(_fval, 42.0)

    def test_compare_par_values_minimize_fcn3(self):
        self.m3.minimize()
        _par_vals = self.m3.parameter_values
        self.assertIs(type(_par_vals), np.ndarray)
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3, _par_vals, rtol=0, atol=1e-6))

    def test_compare_par_errors_minimize_fcn3(self):
        self.m3.minimize()
        _par_errs = self.m3.parameter_errors
        self.assertIs(type(_par_errs), np.ndarray)
        self.assertTrue(np.allclose(
            self._ref_par_err_fcn3, _par_errs, rtol=0, atol=1e-6))

    def test_compare_func_value_minimize_fcn3(self):
        self.m3.minimize()
        _fval = self.m3.function_value
        self.assertTrue(type(_fval) in [float, np.float64])
        self.assertAlmostEqual(_fval, -5.23)

    def test_compare_par_values_minimize_fcn3_fix_release_x(self):
        self.m3.fix('x')
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3_fix_x, self.m3.parameter_values, rtol=0, atol=1e-6))

        self.m3.release('x')
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3, self.m3.parameter_values, rtol=0, atol=1e-6))

    def test_compare_par_errors_minimize_fcn3_fix_release_x(self):
        self.m3.fix('x')
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_err_fcn3_fix_x, self.m3.parameter_errors, rtol=0, atol=1e-6))

        self.m3.release('x')
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_err_fcn3, self.m3.parameter_errors, rtol=0, atol=1e-6))

    def test_compare_par_values_minimize_fcn3_limit_unlimit_x(self):
        self.m3.limit('x', (-1., 0.7))
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3_limit_x, self.m3.parameter_values, rtol=0, atol=1e-6))

        self.m3.unlimit('x')
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3, self.m3.parameter_values, rtol=0, atol=1e-6))

    def test_compare_par_values_minimize_fcn3_limit_unlimit_onesided(self):
        self.m3.limit('x', (None, 0.7))
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3_limit_x, self.m3.parameter_values, rtol=0, atol=1e-6))

        self.m3.unlimit('x')
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3, self.m3.parameter_values, rtol=0, atol=1e-6))

    def test_compare_par_values_minimize_fcn3_limit_unlimit_onesided_2(self):
        self.m3.limit('x', (3.5, None))
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3_limit_x_2, self.m3.parameter_values, rtol=0, atol=1e-6))

        self.m3.unlimit('x')
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3, self.m3.parameter_values, rtol=0, atol=1e-6))

    def test_limit_x_after_minimize_fcn3(self):
        self.m3.minimize()
        self.m3.limit("x", (-1.0, 0.7))
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3_limit_x, self.m3.parameter_values, rtol=0, atol=1e-6))

    def test_limit_x_after_minimize_fcn3_2(self):
        self.m3.minimize()
        self.m3.limit("x", (3.5, 10.0))
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3_limit_x_2, self.m3.parameter_values, rtol=0, atol=1e-6))

    def test_compare_par_values_minimize_fcn3_fix_x_limit_y(self):
        self.m3.fix("x")
        self.m3.limit("y", (-10, 0.5))
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3_fix_x_limit_y, self.m3.parameter_values, rtol=0, atol=1e-6))

        self.m3.release("x")
        self.m3.unlimit("y")
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self._ref_par_val_fcn3, self.m3.parameter_values, rtol=0, atol=1e-6))

    def test_compare_cov_mat_minimize_fcn3(self):
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.cov_mat, self._ref_cov_mat_fcn3, rtol=0, atol=1e-6)
        )
        self.assertAlmostEqual(self.m3.function_value, self._ref_fval_fcn3)
        self.assertTrue(np.allclose(self.m3.parameter_values, self._ref_par_val_fcn3))

    def test_compare_cor_mat_minimize_fcn3(self):
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.cor_mat, self._ref_cor_mat_fcn3, rtol=0, atol=1e-6)
        )
        self.assertAlmostEqual(self.m3.function_value, self._ref_fval_fcn3)
        self.assertTrue(np.allclose(self.m3.parameter_values, self._ref_par_val_fcn3))

    def test_compare_cov_mat_minimize_fcn3_fix_x(self):
        self.m3.fix('x')
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.cov_mat, self._ref_cov_mat_fcn3_fix_x, rtol=0, atol=1e-6)
        )

    def test_compare_cor_mat_minimize_fcn3_fix_x(self):
        self.m3.fix('x')
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.cor_mat, self._ref_cor_mat_fcn3_fix_x, rtol=0, atol=1e-6)
        )

    def test_compare_cov_mat_minimize_fcn3_fix_x_reverse(self):
        self.m3.minimize()
        self.m3.fix('x')
        self.assertTrue(
            np.allclose(self.m3.cov_mat, self._ref_cov_mat_fcn3_fix_x, rtol=0, atol=1e-6)
        )

    def test_compare_cor_mat_minimize_fcn3_fix_x_reverse(self):
        self.m3.minimize()
        self.m3.fix('x')
        self.assertTrue(
            np.allclose(self.m3.cor_mat, self._ref_cor_mat_fcn3_fix_x, rtol=0, atol=1e-6)
        )

    def test_raise_fix_xyz(self):
        self.m3.fix_several(["x", "y", "z"])
        with self.assertRaises(MinimizerException):
            self.m3.minimize()

    def test_compare_par_values_minimize_fcn1_errdef_4(self):
        self.m1.errordef = 4.0
        self.m1.minimize()
        self.assertAlmostEqual(self.m1.parameter_values[0], 42.)

    def test_compare_par_errors_minimize_fcn1_errdef_4(self):
        self.m1.errordef = 4.0
        self.m1.minimize()
        self.assertAlmostEqual(self.m1.parameter_errors[0], np.sqrt(4.0 * (1. / 42.)))

    def test_compare_cov_mat_minimize_fcn3_errdef_4(self):
        self.m3.errordef = 4.0
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.cov_mat, 4.0 * self._ref_cov_mat_fcn3, rtol=0, atol=1e-6)
        )

    def test_compare_hessian_minimize_fcn3(self):
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.hessian, self._ref_hessian_fcn3, rtol=0, atol=1e-6)
        )
        self.assertAlmostEqual(self.m3.function_value, self._ref_fval_fcn3)
        self.assertTrue(np.allclose(self.m3.parameter_values, self._ref_par_val_fcn3))

    def test_compare_hessian_minimize_fcn3_fix_x(self):
        self.m3.fix("x")
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.hessian, self._ref_hessian_fcn3_fix_x, rtol=0, atol=1e-6)
        )

    def test_compare_hessian_minimize_fcn3_fix_x_reverse(self):
        self.m3.minimize()
        self.m3.fix("x")
        self.assertTrue(
            np.allclose(self.m3.hessian, self._ref_hessian_fcn3_fix_x, rtol=0, atol=1e-6)
        )

    def test_compare_hessian_minimize_fcn3_errdef_4(self):
        self.m3.errordef = 4.0
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.hessian, self._ref_hessian_fcn3, rtol=0, atol=1e-6)
        )

    def test_compare_hessian_inv_minimize_fcn3(self):
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.hessian_inv, self._ref_hessian_inv_fcn3, rtol=0, atol=1e-6)
        )
        self.assertAlmostEqual(self.m3.function_value, self._ref_fval_fcn3)
        self.assertTrue(np.allclose(self.m3.parameter_values, self._ref_par_val_fcn3))

    def test_compare_hessian_inv_minimize_fcn3_fix_x(self):
        self.m3.fix("x")
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.hessian_inv, self._ref_hessian_inv_fcn3_fix_x, rtol=0, atol=1e-6)
        )

    def test_compare_hessian_inv_minimize_fcn3_fix_x_reverse(self):
        self.m3.minimize()
        self.m3.fix("x")
        self.assertTrue(
            np.allclose(self.m3.hessian_inv, self._ref_hessian_inv_fcn3_fix_x, rtol=0, atol=1e-6)
        )

    def test_compare_hessian_inv_minimize_fcn3_errdef_4(self):
        self.m3.errordef = 4.0
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.hessian_inv, self._ref_hessian_inv_fcn3, rtol=0, atol=1e-6)
        )

    def test_compare_asymm_errs_minimize_fnc1(self):
        self.m1.minimize()
        self.assertAlmostEqual(self.m1.asymmetric_parameter_errors[0, 0], -np.sqrt(1.0 / 42.0))
        self.assertAlmostEqual(self.m1.asymmetric_parameter_errors[0, 1], np.sqrt(1.0 / 42.0))
        self.assertAlmostEqual(self.m1.function_value, 42)
        self.assertAlmostEqual(self.m1.parameter_values[0], 42.)

    def test_compare_asymm_errs_minimize_fnc3(self):
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self.m3.asymmetric_parameter_errors[:, 0], -self._ref_par_err_fcn3))
        self.assertTrue(np.allclose(
            self.m3.asymmetric_parameter_errors[:, 1], self._ref_par_err_fcn3))
        self.assertAlmostEqual(self.m3.function_value, self._ref_fval_fcn3)
        self.assertTrue(np.allclose(self.m3.parameter_values, self._ref_par_val_fcn3))

    def test_compare_asymm_errs_minimize_fnc3_fix_x(self):
        self.m3.fix("x")
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self.m3.asymmetric_parameter_errors[:, 0], -self._ref_par_err_fcn3_fix_x))
        self.assertTrue(np.allclose(
            self.m3.asymmetric_parameter_errors[:, 1], self._ref_par_err_fcn3_fix_x))
        self.assertTrue(np.allclose(self.m3.parameter_values, self._ref_par_val_fcn3_fix_x))

    def test_compare_asymm_errs_minimize_fnc3_fix_x_reverse(self):
        self.m3.minimize()
        self.m3.fix("x")
        self.assertTrue(np.allclose(
            self.m3.asymmetric_parameter_errors[:, 0], -self._ref_par_err_fcn3_fix_x))
        self.assertTrue(np.allclose(
            self.m3.asymmetric_parameter_errors[:, 1], self._ref_par_err_fcn3_fix_x))
        self.assertTrue(np.allclose(self.m3.parameter_values, self._ref_par_val_fcn3))

    def test_compare_par_values_to_scipy_optimize(self):
        self.m3.minimize()
        self.assertTrue(
            np.allclose(self.m3.parameter_values, self._scipy_fmin.x, rtol=0, atol=1e-6)
        )

    @unittest.skip("SLSQP does not compute hessian by default")
    def test_compare_hessian_inv_to_scipy_optimize(self):
        self.m3.minimize()
        _hm_inv = self.m3.hessian_inv
        _hm_inv_scipy = self._scipy_fmin.hess_inv
        self.assertTrue(
            np.allclose(_hm_inv, _hm_inv_scipy, atol=1e-2)
        )

    def test_compare_func_value_set_fcn3_parameters(self):
        self.m3.set('x', 1.23)
        self.m3.set('y', 4.32)
        self.m3.set('z', 9.81)
        self.assertAlmostEqual(self.m3.function_value, -5.23)

    def test_set_raise(self):
        for _minimizer in [self.m1, self.m3]:
            with self.assertRaises(ValueError):
                _minimizer.set("DEADBEEF", 1)

    def test_profile_m3_x(self):
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self.m3.profile('x', bins=5, subtract_min=True),
            self._ref_profile_m3_x_5, atol=1e-7
        ))

    def test_profile_m3_x_no_subtract_min(self):
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self.m3.profile('x', bins=5, subtract_min=False),
            self._ref_profile_m3_x_5_no_subtract_min, atol=1e-7
        ))

    def test_profile_m3_x_fix_y(self):
        self.m3.fix("y")
        self.m3.minimize()
        self.assertTrue(np.allclose(
            self.m3.profile('x', bins=5, subtract_min=True),
            self._ref_profile_m3_x_5, atol=1e-7
        ))

    def test_profile_raise_no_fit(self):
        with self.assertRaises(MinimizerException):
            self.m3.profile("x")

    @unittest.skip("Testing contours not yet implemented!")
    def test_contour_m3_x_y(self):
        self.m3.minimize()
        _cont = self.m3.contour('x', 'y', numpoints=5, sigma=1.0)
        self.assertTrue(
            np.allclose(_cont, self._ref_contour_m3_x_y_5, atol=1e-3)
        )

    def test_contour_raise_no_fit(self):
        with self.assertRaises(MinimizerException):
            self.m3.contour("x", "y")
