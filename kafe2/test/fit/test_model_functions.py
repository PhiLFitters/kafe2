#!/usr/bin/env python3

import unittest

import numpy as np

from kafe2.fit._base.model import ModelFunctionBase
from kafe2.fit.histogram.model import HistModelFunction
from kafe2.fit.indexed.model import IndexedModelFunction


class TestSymPy(unittest.TestCase):
    MODEL_FUNCTION_CLASS = ModelFunctionBase

    def setUp(self):
        def model(x, a, b, c):
            return a * x**2 + b * x + c

        self._model_1_python = model
        self._model_1_sympy = "x a b c -> a * x^2 + b * x + c"

        self._ref_properties_1 = dict(
            x=np.arange(10),
            parameter_values=[1.2, 3.4, 5.6],
            name="model",
            parameter_names=["a", "b", "c"],
            argcount=4,
            parcount=3,
            x_name=["x"],
            defaults=[1.0, 1.0, 1.0],
            latex_name=r"{\tt model}",
            parameter_latex_names=["{a}", "{b}", "{c}"],
        )
        _x = self._ref_properties_1["x"]
        _pvs = self._ref_properties_1["parameter_values"]
        self._ref_properties_1["y"] = _pvs[0] * _x**2 + _pvs[1] * _x + _pvs[2]

        def other_model(tau, A_0=3.0, tau_0=1.0):
            return A_0 * np.exp(-tau / tau_0)

        self._model_2_python = other_model
        self._model_2_sympy = "other_model: tau A_0=3.0 tau_0 -> A_0 * exp(-tau / tau_0)"

        self._ref_properties_2 = dict(
            x=np.arange(10),
            parameter_values=[1.2, 3.4],
            name="other_model",
            parameter_names=["A_0", "tau_0"],
            argcount=3,
            parcount=2,
            x_name=["tau"],
            defaults=[3.0, 1.0],
            latex_name=r"{\tt other\_model}",
            parameter_latex_names=["{A_0}", r"{\tau_0}"],
        )
        _x = self._ref_properties_2["x"]
        _pvs = self._ref_properties_2["parameter_values"]
        self._ref_properties_2["y"] = _pvs[0] * np.exp(-_x / _pvs[1])

    def _assert_properties_correct(self, model_function, ref_properties):
        with self.subTest("y"):
            self.assertTrue(
                np.allclose(
                    model_function(ref_properties["x"], *ref_properties["parameter_values"]),
                    ref_properties["y"],
                )
            )
        with self.subTest("name"):
            self.assertEqual(model_function.name, ref_properties["name"])
        with self.subTest("parameter_names"):
            self.assertEqual(model_function.parameter_names, ref_properties["parameter_names"])
        with self.subTest("argcount"):
            self.assertEqual(model_function.argcount, ref_properties["argcount"])
        with self.subTest("parcount"):
            self.assertEqual(model_function.parcount, ref_properties["parcount"])
        with self.subTest("x_name"):
            self.assertEqual(model_function.x_name, ref_properties["x_name"])
        with self.subTest("defaults"):
            self.assertEqual(model_function.defaults, ref_properties["defaults"])
        with self.subTest("formatter_name"):
            self.assertEqual(model_function.formatter.name, ref_properties["name"])
        with self.subTest("formatter_latex_name"):
            self.assertEqual(model_function.formatter.latex_name, ref_properties["latex_name"])
        with self.subTest("formatter_parameter_names"):
            self.assertEqual(
                [_pf.name for _pf in model_function.formatter.par_formatters],
                ref_properties["parameter_names"],
            )
        with self.subTest("formatter_parameter_latex_names"):
            self.assertEqual(
                [_pf.latex_name for _pf in model_function.formatter.par_formatters],
                ref_properties["parameter_latex_names"],
            )

    def test_properties_python_1(self):
        model_function = self.MODEL_FUNCTION_CLASS(self._model_1_python)
        self._assert_properties_correct(model_function, self._ref_properties_1)

    def test_properties_sympy_1(self):
        model_function = self.MODEL_FUNCTION_CLASS(self._model_1_sympy)
        self._assert_properties_correct(model_function, self._ref_properties_1)

    def test_properties_python_2(self):
        model_function = self.MODEL_FUNCTION_CLASS(self._model_2_python)
        self._assert_properties_correct(model_function, self._ref_properties_2)

    def test_properties_sympy_2(self):
        model_function = self.MODEL_FUNCTION_CLASS(self._model_2_sympy)
        self._assert_properties_correct(model_function, self._ref_properties_2)


class TestSymPyHist(TestSymPy):
    MODEL_FUNCTION_CLASS = HistModelFunction


class TestSymPyIndexed(TestSymPy):
    MODEL_FUNCTION_CLASS = IndexedModelFunction

    def setUp(self):
        super().setUp()
        self._ref_properties_1["parameter_names"] = self._ref_properties_1["x_name"] + self._ref_properties_1["parameter_names"]
        self._ref_properties_1["parcount"] += 1
        self._ref_properties_1["x_name"] = []
        self._ref_properties_1["defaults"].insert(0, 1.0)
        self._ref_properties_1["parameter_latex_names"].insert(0, r"{x}")

        self._ref_properties_2["parameter_names"] = self._ref_properties_2["x_name"] + self._ref_properties_2["parameter_names"]
        self._ref_properties_2["parcount"] += 1
        self._ref_properties_2["x_name"] = []
        self._ref_properties_2["defaults"].insert(0, 1.0)
        self._ref_properties_2["parameter_latex_names"].insert(0, r"{\tau}")
