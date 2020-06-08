import numpy as np
import unittest2 as unittest
from scipy.special import factorial

from kafe2.core.constraint import GaussianSimpleParameterConstraint, \
    GaussianMatrixParameterConstraint
from kafe2.fit._base.cost import *
from kafe2.fit.histogram.cost import *
from kafe2.fit.indexed.cost import *
from kafe2.fit.xy.cost import *


class TestCostBuiltin(unittest.TestCase):
    CHI2_COST_FUNCTION = CostFunction_Chi2
    NLL_COST_FUNCTION = CostFunction_NegLogLikelihood

    def setUp(self):
        self._data_chi2 = np.array([-0.5, 2.1, 8.9])
        self._model_chi2 = np.array([5.7, 8.4, -2.3])
        self._data_poisson = np.array([0.0, 2.0, 9.0])
        self._model_poisson = np.array([5.7, 8.4, 2.3])
        self._res = self._data_chi2 - self._model_chi2
        self._cov_mat = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 2.0, 0.3],
            [0.2, 0.3, 3.0]
        ])
        self._cov_mat_inv = np.linalg.inv(self._cov_mat)
        self._pointwise_errors = np.sqrt(np.diag(self._cov_mat))

        self._cost_chi2_cov_mat = self._res.dot(self._cov_mat_inv).dot(self._res)
        self._cost_chi2_pointwise = np.sum((self._res / self._pointwise_errors) ** 2)
        self._cost_chi2_no_errors = np.sum(self._res ** 2)
        self._cost_nll_gaussian = self._cost_chi2_pointwise + 2.0 * np.sum(np.log(
            np.sqrt((2.0 * np.pi)) * self._pointwise_errors))
        self._cost_nll_poisson = -2.0 * np.sum(
            np.log(self._model_poisson ** self._data_poisson)
            - np.log(factorial(self._data_poisson)) - self._model_poisson)
        self._cost_nllr_poisson = self._cost_nll_poisson + 2.0 * np.sum(
            np.log(self._data_poisson ** self._data_poisson)
            - np.log(factorial(self._data_poisson)) - self._data_poisson)
        self._par_vals = np.array([12.3, 0.001, -1.9])
        self._simple_constraint = GaussianSimpleParameterConstraint(
            index=0, value=10.0, uncertainty=2.5)
        self._matrix_constraint = GaussianMatrixParameterConstraint(
            indices=(0, 1, 2),
            values=(1.0, 2.0, 3.0),
            matrix=[
                [1.5, 0.1, 0.1],
                [0.1, 2.2, 0.1],
                [0.1, 0.1, 0.3]
            ]
        )
        self. _par_constraints = [self._simple_constraint, self._matrix_constraint]
        self._par_cost = self._simple_constraint.cost(self._par_vals) \
            + self._matrix_constraint.cost(self._par_vals)

    def test_chi2_no_errors(self):
        self.assertAlmostEqual(
            self._cost_chi2_no_errors,
            self.CHI2_COST_FUNCTION(errors_to_use=None)
            (self._data_chi2, self._model_chi2, None, None))
        self.assertAlmostEqual(
            self._cost_chi2_no_errors + self._par_cost,
            self.CHI2_COST_FUNCTION(errors_to_use=None)
            (self._data_chi2, self._model_chi2, self._par_vals, self._par_constraints))

    def test_chi2_pointwise(self):
        self.assertAlmostEqual(
            self._cost_chi2_pointwise,
            self.CHI2_COST_FUNCTION(errors_to_use='pointwise')
            (self._data_chi2, self._model_chi2, self._pointwise_errors, None, None))
        self.assertAlmostEqual(
            self._cost_chi2_pointwise + self._par_cost,
            self.CHI2_COST_FUNCTION(errors_to_use='pointwise')
            (self._data_chi2, self._model_chi2, self._pointwise_errors,
             self._par_vals, self._par_constraints))

    def test_chi2_cov_mat(self):
        self.assertAlmostEqual(
            self._cost_chi2_cov_mat,
            self.CHI2_COST_FUNCTION(errors_to_use='covariance')
            (self._data_chi2, self._model_chi2, self._cov_mat_inv, None, None))
        self.assertAlmostEqual(
            self._cost_chi2_cov_mat + self._par_cost,
            self.CHI2_COST_FUNCTION(errors_to_use='covariance')
            (self._data_chi2, self._model_chi2, self._cov_mat_inv,
             self._par_vals, self._par_constraints))

    def test_nll_gaussian(self):
        self.assertAlmostEqual(
            self._cost_nll_gaussian,
            self.NLL_COST_FUNCTION(data_point_distribution='gaussian')
            (self._data_chi2, self._model_chi2, self._pointwise_errors, None, None))
        self.assertAlmostEqual(
            self._cost_nll_gaussian + self._par_cost,
            self.NLL_COST_FUNCTION(data_point_distribution='gaussian')
            (self._data_chi2, self._model_chi2, self._pointwise_errors,
             self._par_vals, self._par_constraints))

    def test_nll_poisson(self):
        self.assertAlmostEqual(
            self._cost_nll_poisson,
            self.NLL_COST_FUNCTION(data_point_distribution='poisson')
            (self._data_poisson, self._model_poisson, None, None))
        self.assertAlmostEqual(
            self._cost_nll_poisson + self._par_cost,
            self.NLL_COST_FUNCTION(data_point_distribution='poisson')
            (self._data_poisson, self._model_poisson, self._par_vals, self._par_constraints))

    def test_nllr_gaussian(self):
        self.assertAlmostEqual(
            self._cost_chi2_pointwise,
            self.NLL_COST_FUNCTION(data_point_distribution='gaussian', ratio=True)
            (self._data_chi2, self._model_chi2, self._pointwise_errors, None, None))
        self.assertAlmostEqual(
            self._cost_chi2_pointwise + self._par_cost,
            self.NLL_COST_FUNCTION(data_point_distribution='gaussian', ratio=True)
            (self._data_chi2, self._model_chi2, self._pointwise_errors,
             self._par_vals, self._par_constraints))

    def test_nllr_poisson(self):
        self.assertAlmostEqual(
            self._cost_nllr_poisson,
            self.NLL_COST_FUNCTION(data_point_distribution='poisson', ratio=True)
            (self._data_poisson, self._model_poisson, None, None))
        self.assertAlmostEqual(
            self._cost_nllr_poisson + self._par_cost,
            self.NLL_COST_FUNCTION(data_point_distribution='poisson', ratio=True)
            (self._data_poisson, self._model_poisson, self._par_vals, self._par_constraints))

    def test_chi2_raise(self):
        with self.assertRaises(ValueError):
            self.CHI2_COST_FUNCTION(errors_to_use="XYZ")
        with self.assertRaises(ValueError):
            self.CHI2_COST_FUNCTION(errors_to_use="covariance")(
                self._data_chi2, np.ones(10), self._cov_mat_inv, None, None)
        with self.assertRaises(CostFunctionException):
            self.CHI2_COST_FUNCTION(errors_to_use="covariance", fallback_on_singular=False)(
                self._data_chi2, self._model_chi2, None, None, None)
        with self.assertRaises(CostFunctionException):
            self.CHI2_COST_FUNCTION(errors_to_use="pointwise", fallback_on_singular=False)(
                self._data_chi2, self._model_chi2, np.arange(len(self._pointwise_errors)),
                None, None)

    def test_nll_raise(self):
        with self.assertRaises(ValueError):
            self.NLL_COST_FUNCTION(data_point_distribution="yes")

    def test_inf_cost(self):
        self.assertEqual(
            np.inf,
            self.CHI2_COST_FUNCTION(errors_to_use="covariance")(
                self._data_chi2, np.nan * np.ones_like(self._model_chi2), self._cov_mat_inv,
                None, None)
        )
        self.assertEqual(
            np.inf,
            self.CHI2_COST_FUNCTION(errors_to_use="pointwise")(
                self._data_chi2, np.nan * np.ones_like(self._model_chi2), self._pointwise_errors,
                None, None)
        )
        self.assertEqual(
            np.inf,
            self.CHI2_COST_FUNCTION(errors_to_use=None)(
                self._data_chi2, np.nan * np.ones_like(self._model_chi2), None, None)
        )
        self.assertEqual(
            np.inf,
            self.NLL_COST_FUNCTION("poisson", ratio=False)(
                self._data_poisson, -self._model_poisson, None, None)
        )
        self.assertEqual(
            np.inf,
            self.NLL_COST_FUNCTION("poisson", ratio=True)(
                self._data_poisson, -self._model_poisson, None, None)
        )
        self.assertEqual(
            np.inf,
            self.NLL_COST_FUNCTION("gaussian", ratio=False)(
                self._data_chi2, self._model_chi2, -self._pointwise_errors, None, None)
        )
        self.assertEqual(
            np.inf,
            self.NLL_COST_FUNCTION("gaussian", ratio=True)(
                self._data_chi2, self._model_chi2, -self._pointwise_errors, None, None)
        )


class TestCostBuiltinHist(TestCostBuiltin):

    CHI2_COST_FUNCTION = HistCostFunction_Chi2
    NLL_COST_FUNCTION = HistCostFunction_NegLogLikelihood


class TestCostBuiltinIndexed(TestCostBuiltin):

    CHI2_COST_FUNCTION = IndexedCostFunction_Chi2
    NLL_COST_FUNCTION = IndexedCostFunction_NegLogLikelihood


class TestCostBuiltinXY(TestCostBuiltin):

    CHI2_COST_FUNCTION = XYCostFunction_Chi2
    NLL_COST_FUNCTION = XYCostFunction_NegLogLikelihood


class TestCostUserDefined(unittest.TestCase):
    def setUp(self):
        def my_cost(a, b, c, d):
            return a ** 2 + (b + c) ** 2 + (d - 1) ** 2

        def my_cost_varargs(*args):
            return args[0] ** 2 + (args[1] + args[2]) ** 2 + (args[3] - 1) ** 2

        self._ref_par_vals = [1, 2, 5, 7]
        self._ref_cost = my_cost(*self._ref_par_vals)
        self._constraint = GaussianSimpleParameterConstraint(index=0, value=0, uncertainty=1)
        self._matrix_constraint = GaussianMatrixParameterConstraint(
            indices=[1], values=[0], matrix=[[1]])
        self._constraint_cost = self._ref_par_vals[0] ** 2 + self._ref_par_vals[1] ** 2
        self._ref_par_vals_constraints = self._ref_par_vals + [
            self._ref_par_vals, [self._constraint, self._matrix_constraint]]

        self._cost_func = CostFunction(my_cost, arg_names=None, add_constraint_cost=False)
        self._cost_func_constraints = CostFunction(
            my_cost, arg_names=None, add_constraint_cost=True)
        self._cost_func_varargs = CostFunction(
            my_cost_varargs, arg_names=["a", "b", "c", "d"], add_constraint_cost=False)
        self._cost_func_varargs_constraints = CostFunction(
            my_cost_varargs, arg_names=["a", "b", "c", "d"], add_constraint_cost=True)

    def test_properties(self):
        self.assertEqual(self._cost_func.name, "my_cost")
        self.assertEqual(self._cost_func.arg_names, ["a", "b", "c", "d"])
        self.assertEqual(self._cost_func_varargs.name, "my_cost_varargs")
        self.assertEqual(self._cost_func.arg_names, ["a", "b", "c", "d"])

    def test_validate_raise(self):
        def _cost_args(*args):
            return np.sum(args)

        def _cost_kwargs(**kwargs):
            return np.sum(list(kwargs.values()))

        def _cost_bad_arg_name(cost):
            return cost

        _ = CostFunction(_cost_args, arg_names=["a", "b"])
        with self.assertRaises(CostFunctionException):
            _ = CostFunction(_cost_args)
        with self.assertRaises(CostFunctionException):
            _ = CostFunction(_cost_kwargs)
        with self.assertRaises(CostFunctionException):
            _ = CostFunction(_cost_kwargs, arg_names=["a", "b"])
        with self.assertRaises(CostFunctionException):
            _ = CostFunction(_cost_bad_arg_name)
        _ = CostFunction(_cost_bad_arg_name, arg_names=["a"])
        with self.assertRaises(CostFunctionException):
            _ = CostFunction(_cost_args, arg_names=["cost"])

    def test_compare_cost(self):
        self.assertEqual(
            self._ref_cost, self._cost_func(*self._ref_par_vals)
        )
        self.assertEqual(
            self._ref_cost, self._cost_func_constraints(*(self._ref_par_vals + [None, None]))
        )
        self.assertEqual(
            self._ref_cost,
            self._cost_func_constraints(*(self._ref_par_vals + [self._ref_par_vals, None]))
        )
        self.assertEqual(
            self._ref_cost + self._constraint_cost,
            self._cost_func_constraints(*self._ref_par_vals_constraints)
        )

    def test_compare_cost_varargs(self):
        self.assertEqual(
            self._ref_cost, self._cost_func_varargs(*self._ref_par_vals)
        )
        self.assertEqual(
            self._ref_cost, self._cost_func_varargs_constraints(*(
                    self._ref_par_vals + [None, None]))
        )
        self.assertEqual(
            self._ref_cost,
            self._cost_func_varargs_constraints(*(
                    self._ref_par_vals + [self._ref_par_vals, None]))
        )
        self.assertEqual(
            self._ref_cost + self._constraint_cost,
            self._cost_func_varargs_constraints(*self._ref_par_vals_constraints)
        )

