#!/usr/bin/env python3

import unittest
import warnings

import numpy as np

from kafe2.fit.xy.fit import XYFit


def model_xy(x, par_a, par_b=1.0):
    return par_a * x + par_b


class FitWarningsTest(unittest.TestCase):
    def setUp(self):
        self._x_data = np.linspace(start=-10, stop=10, num=21)
        self._y_data = self._x_data + 1
        self._fit = XYFit([self._x_data, self._y_data], model_function=model_xy)
        self._fit.add_error("y", 0.1)

    def test_bad_defaults_warning(self):
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
