#!/usr/bin/env python3

import os
import shutil
import unittest

import numpy as np

from kafe2 import XYFit, function_library, xy_fit


class TestWrapperCallableXY(unittest.TestCase):
    def _assert_results_equal(self, results_1, results_2):
        self.assertEqual(results_1["did_fit"], results_2["did_fit"])
        self.assertAlmostEqual(results_1["cost"], results_2["cost"])
        self.assertAlmostEqual(results_1["goodness_of_fit"], results_2["goodness_of_fit"])
        self.assertAlmostEqual(results_1["chi2_probability"], results_2["chi2_probability"])
        self.assertAlmostEqual(results_1["gof/ndf"], results_2["gof/ndf"])
        self.assertEqual(results_1["ndf"], results_2["ndf"])
        self.assertTrue(
            np.allclose(
                list(results_1["parameter_values"].values()),
                list(results_2["parameter_values"].values()),
            )
        )
        self.assertTrue(
            np.allclose(
                list(results_1["parameter_errors"].values()),
                list(results_2["parameter_errors"].values()),
            )
        )
        self.assertTrue(np.allclose(results_1["parameter_cor_mat"], results_2["parameter_cor_mat"], rtol=1e-4))
        self.assertTrue(np.allclose(results_1["parameter_cov_mat"], results_2["parameter_cov_mat"], rtol=1e-4))
        try:
            self.assertTrue(
                np.allclose(
                    list(results_1["asymmetric_parameter_errors"].values()),
                    list(results_2["asymmetric_parameter_errors"].values()),
                )
            )
        except AttributeError:
            self.assertEqual(results_1["asymmetric_parameter_errors"], None)
            self.assertEqual(results_2["asymmetric_parameter_errors"], None)

    def _assert_wrapper_equal(
        self,
        model_function=function_library.linear_model,
        p0=None,
        dp0=None,
        rel_to_model=True,
        limits=None,
        constraints=None,
        profile=None,
        save=True,
        test_matrix=False,
    ):
        _fit = XYFit([self._x_data, self._y_data], model_function)
        _error_ref = "model" if rel_to_model else "data"
        if p0 is not None:
            _fit.set_all_parameter_values(p0)
        if dp0 is not None:
            _fit.parameter_errors = dp0
        _fit.add_error("x", self._x_error)
        _fit.add_error("y", self._y_error)
        _fit.add_error("x", self._x_error_rel, relative=True)
        _fit.add_error("y", self._y_error_rel, relative=True, reference=_error_ref)
        _fit.add_error("x", self._x_error_cor, correlation=1.0)
        _fit.add_error("y", self._y_error_cor[0], correlation=1.0)
        _fit.add_error("y", self._y_error_cor[1], correlation=1.0)
        _fit.add_error("x", self._x_error_cor_rel[0], correlation=1.0, relative=True)
        _fit.add_error("x", self._x_error_cor_rel[1], correlation=1.0, relative=True)
        _fit.add_error("y", self._y_error_cor_rel, correlation=1.0, relative=True, reference=_error_ref)
        if limits is not None:
            _limits_manual = [limits] if isinstance(limits[0], str) else limits
            for _limit in _limits_manual:
                _fit.limit_parameter(*_limit)
        if constraints is not None:
            _constraints_manual = [constraints] if isinstance(constraints[0], str) else constraints
            for _constraint in _constraints_manual:
                _fit.add_parameter_constraint(*_constraint)
        _profile_manual = profile is None or profile
        _result_1 = _fit.do_fit(asymmetric_parameter_errors=_profile_manual)

        if test_matrix:
            self._x_error = np.eye(10) * self._x_error**2
            self._x_error += np.sum(np.square(self._x_error_cor))
            self._x_error_cor = None
            self._y_error = np.eye(10) * self._y_error**2
            self._y_error += np.sum(np.square(self._y_error_cor))
            self._y_error_cor = None

        _result_2 = xy_fit(
            model_function,
            self._x_data,
            self._y_data,
            p0,
            dp0,
            x_error=self._x_error,
            y_error=self._y_error,
            x_error_rel=self._x_error_rel,
            y_error_rel=self._y_error_rel,
            x_error_cor=self._x_error_cor,
            y_error_cor=self._y_error_cor,
            x_error_cor_rel=self._x_error_cor_rel,
            y_error_cor_rel=self._y_error_cor_rel,
            errors_rel_to_model=rel_to_model,
            limits=limits,
            constraints=constraints,
            profile=profile,
            save=save,
        )
        self._assert_results_equal(_result_1, _result_2)

    def setUp(self):
        np.random.seed(12345)
        self._true_pars = [1.2, 3.4]

        self._x_error = 0.1
        self._y_error = np.array([0.1] * 5 + [0.2] * 5)
        self._x_error_rel = np.array([0.02] * 5 + [0.0] * 5)
        self._y_error_rel = 0.02
        self._x_error_cor = 0.05
        self._y_error_cor = [0.05, 0.1]
        self._x_error_cor_rel = [0.01, 0.02]
        self._y_error_cor_rel = 0.01

        self._x_data = np.arange(10)
        self._y_data = function_library.linear_model(self._x_data + 0.2 * np.random.randn(10), *self._true_pars) + 0.5 * np.random.randn(10)

    def tearDown(self):
        if os.path.exists("results"):
            shutil.rmtree("results")

    def test_default_model_function(self):
        self._assert_wrapper_equal()

    def test_quadratic_model_function(self):
        self._assert_wrapper_equal(function_library.quadratic_model)

    def test_set_p0(self):
        self._assert_wrapper_equal(function_library.exponential_model, p0=[2.0, 2.0], dp0=[0.01, 0.01])

    def test_rel_to_data(self):
        self._assert_wrapper_equal(rel_to_model=False)

    def test_limits(self):
        self._assert_wrapper_equal(limits=["a", -2, 2])
        self._assert_wrapper_equal(limits=[["a", -2, 2], ["b", -2, 2]])

    def test_constraints(self):
        self._assert_wrapper_equal(constraints=["a", -2, 2])
        self._assert_wrapper_equal(constraints=[["a", -2, 2], ["b", -2, 2]])

    def test_explicit_profile(self):
        self._assert_wrapper_equal(profile=False)
        self._assert_wrapper_equal(profile=True)

    def test_matrix(self):
        self._assert_wrapper_equal(test_matrix=True)

    def test_save(self):
        self._assert_wrapper_equal()
        self.assertEqual(len(os.listdir("results")), 2)
        self._assert_wrapper_equal(save=False)
        self.assertEqual(len(os.listdir("results")), 2)
        self._assert_wrapper_equal()
        self.assertEqual(len(os.listdir("results")), 4)
