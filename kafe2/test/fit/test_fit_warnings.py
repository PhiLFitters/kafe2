#!/usr/bin/env python3

import unittest
import warnings

import numpy as np

from kafe2.fit.xy.fit import XYFit


def model_bad_defaults(x, par_a, par_b=1.0):
    return par_a * x + par_b


class FitWarningsBadDefaultsTest(unittest.TestCase):
    def setUp(self):
        self._x_data = np.linspace(start=-10, stop=10, num=21)
        self._y_data = self._x_data + 1
        self._fit = XYFit([self._x_data, self._y_data], model_function=model_bad_defaults)
        self._fit.add_error("y", 0.1)

    def test_bad_defaults_warning(self):
        with self.assertWarnsRegex(Warning, ".*par_a.*"):
            self._fit.do_fit()

    def test_bad_defaults_warning_zero(self):
        self._fit.set_parameter_values(par_a=0.0)
        with self.assertWarnsRegex(Warning, ".*par_a.*"):
            self._fit.do_fit()

    def test_bad_defaults_warning_set_par_vals(self):
        self._fit.set_parameter_values(par_a=1.0)
        with warnings.catch_warnings(record=True) as w:
            self._fit.do_fit()
        self.assertFalse(w)

    def test_bad_defaults_warning_set_all_par_vals(self):
        self._fit.set_all_parameter_values([1.0, 1.0])
        with warnings.catch_warnings(record=True) as w:
            self._fit.do_fit()
        self.assertFalse(w)

    def test_bad_defaults_warning_set_par_errs(self):
        self._fit.parameter_errors = [0.1, 0.1]
        with warnings.catch_warnings(record=True) as w:
            self._fit.do_fit()
        self.assertFalse(w)

    def test_bad_defaults_warning_fix_par(self):
        self._fit.fix_parameter("par_a", 1.0)
        with warnings.catch_warnings(record=True) as w:
            self._fit.do_fit()
        self.assertFalse(w)

    def test_bad_defaults_warning_limit_par(self):
        self._fit.limit_parameter("par_a", lower=0.0)
        with warnings.catch_warnings(record=True) as w:
            self._fit.do_fit()
        self.assertFalse(w)

    def test_bad_defaults_warning_constrain_par_simple(self):
        self._fit.add_parameter_constraint("par_a", 1.0, 0.1)
        with warnings.catch_warnings(record=True) as w:
            self._fit.do_fit()
        self.assertFalse(w)

    def test_bad_defaults_warning_constrain_par_matrix(self):
        self._fit.add_matrix_parameter_constraint(["par_a", "par_b"], [1.0, 1.0], [[0.01, 0.0001], [0.0001, 0.01]])
        with warnings.catch_warnings(record=True) as w:
            self._fit.do_fit()
        self.assertFalse(w)

    def test_bad_defaults_warning_do_fit(self):
        self._fit.do_fit()
        with warnings.catch_warnings(record=True) as w:
            self._fit.do_fit()
        self.assertFalse(w)


def model_bad_value_range(x, par_a=1.0, par_b=1.0):
    return par_a * x + par_b


def model_bad_value_range_small(x, par_a=1.0, par_b=1.0):
    return 1e-10 * model_bad_value_range(x, par_a, par_b)


def model_bad_value_range_large(x, par_a=1.0, par_b=1.0):
    return 1e10 * model_bad_value_range(x, par_a, par_b)


class FitWarningsBadValueRangeTest(unittest.TestCase):
    def setUp(self):
        self._x_data = np.linspace(start=-1, stop=1, num=3)
        self._y_data = self._x_data + 1

        self._fit = XYFit([self._x_data, self._y_data], model_function=model_bad_value_range)
        self._fit.add_error("y", 0.1)

        self._fit_small = XYFit([self._x_data, self._y_data], model_function=model_bad_value_range_small)
        self._fit_small.add_error("y", 0.1)

        self._fit_large = XYFit([self._x_data, self._y_data], model_function=model_bad_value_range_large)
        self._fit_large.add_error("y", 0.1)

    def test_bad_value_range_data_fine_one(self):
        with warnings.catch_warnings(record=True) as w:
            self._fit.data = [np.ones_like(self._x_data), np.ones_like(self._y_data)]
        self.assertFalse(w)

    def test_bad_value_range_data_fine_zero(self):
        with warnings.catch_warnings(record=True) as w:
            self._fit.data = [np.zeros_like(self._x_data), np.zeros_like(self._y_data)]
        self.assertFalse(w)

    def test_bad_value_range_x_data_small(self):
        with self.assertWarnsRegex(Warning, ".*x_data.*small.*re-scale.*"):
            self._fit.data = [1e-10 * np.ones_like(self._x_data), np.ones_like(self._y_data)]

    def test_bad_value_range_x_data_large(self):
        with self.assertWarnsRegex(Warning, ".*x_data.*large.*re-scale.*"):
            self._fit.data = [1e10 * np.ones_like(self._x_data), np.ones_like(self._y_data)]

    def test_bad_value_range_y_data_small(self):
        with self.assertWarnsRegex(Warning, ".*y_data.*small.*re-scale.*"):
            self._fit.data = [np.ones_like(self._x_data), 1e-10 * np.ones_like(self._y_data)]

    def test_bad_value_range_y_data_large(self):
        with self.assertWarnsRegex(Warning, ".*y_data.*large.*re-scale.*"):
            self._fit.data = [np.ones_like(self._x_data), 1e10 * np.ones_like(self._y_data)]

    def test_bad_value_range_par_small(self):
        with self.assertWarnsRegex(Warning, ".*par_a.*small.*re-scale.*"):
            self._fit.set_parameter_values(par_a=1e-10)

    def test_bad_value_range_par_large(self):
        with self.assertWarnsRegex(Warning, ".*par_a.*large.*re-scale.*"):
            self._fit.set_parameter_values(par_a=1e10)

    def test_bad_value_range_model_small(self):
        with self.assertWarnsRegex(Warning, ".*y_model.*small.*re-scale.*"):
            self._fit_small.do_fit()

    def test_bad_value_range_model_large(self):
        with self.assertWarnsRegex(Warning, ".*y_model.*large.*re-scale.*"):
            self._fit_large.do_fit()
