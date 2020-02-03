import numpy as np
import unittest2 as unittest
from scipy.special import factorial

from kafe2.core.constraint import GaussianSimpleParameterConstraint, GaussianMatrixParameterConstraint
from kafe2.fit._base.cost import *
from kafe2.fit.histogram.cost import *
from kafe2.fit.indexed.cost import *
from kafe2.fit.xy.cost import *


class TestCostBase(unittest.TestCase):
    CHI2_COST_FUNCTION = CostFunctionBase_Chi2
    NLL_COST_FUNCTION = CostFunctionBase_NegLogLikelihood
    NLLR_COST_FUNCTION = CostFunctionBase_NegLogLikelihoodRatio

    def setUp(self):
        self._tol = 1e-10
        self._data = np.array([-0.5, 2.1, 8.9])
        self._model = np.array([5.7, 8.4, -2.3])
        self._data_poisson = np.array([0.0, 2.0, 9.0])
        self._model_poisson = np.array([5.7, 8.4, 2.3])
        self._res = self._data - self._model
        self._cov_mat = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 2.0, 0.3],
            [0.2, 0.3, 3.0]
        ])
        self._cov_mat_inv = np.linalg.inv(self._cov_mat)
        self._pointwise_errors = np.sqrt(np.diag(self._cov_mat))

        self._cost_chi2_cov_mat = self._res.dot(self._cov_mat_inv).dot(self._res)
        self._cost_chi2_pointwise = np.sum((self._res / self._pointwise_errors) ** 2)  # same as nllr_gaussian
        self._cost_chi2_no_errors = np.sum(self._res ** 2)
        self._cost_nll_gaussian = self._cost_chi2_pointwise + 2.0 * np.sum(np.log(np.sqrt((2.0 * np.pi)) * self._pointwise_errors))
        #self._cost_nll_poisson = np.sum(np.log(
        #    self._model ** self._data_rounded / (factorial(self._data_rounded) * np.exp(self._model))))
        self._cost_nll_poisson = -2.0 * np.sum(np.log(self._model_poisson ** self._data_poisson)
                                               - np.log(factorial(self._data_poisson)) - self._model_poisson)
        self._cost_nllr_poisson = self._cost_nll_poisson + 2.0 * np.sum(
            np.log(self._data_poisson ** self._data_poisson)
            - np.log(factorial(self._data_poisson)) - self._data_poisson)
        self._par_vals = np.array([12.3, 0.001, -1.9])
        self._simple_constraint = GaussianSimpleParameterConstraint(index=0, value=10.0, uncertainty=2.5)
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
        self._par_cost = self._simple_constraint.cost(self._par_vals) + self._matrix_constraint.cost(self._par_vals)

    def test_chi2_no_errors(self):
        self.assertTrue(np.abs(
            self._cost_chi2_no_errors - self.CHI2_COST_FUNCTION(errors_to_use=None)
            (self._data, self._model, None, None)) < self._tol)
        self.assertTrue(np.abs(
            self._cost_chi2_no_errors + self._par_cost - self.CHI2_COST_FUNCTION(errors_to_use=None)
            (self._data, self._model, self._par_vals, self._par_constraints)) < self._tol)

    def test_chi2_pointwise(self):
        self.assertTrue(np.abs(
            self._cost_chi2_pointwise - self.CHI2_COST_FUNCTION(errors_to_use='pointwise')
            (self._data, self._model, self._pointwise_errors, None, None)) < self._tol)
        self.assertTrue(np.abs(
            self._cost_chi2_pointwise + self._par_cost - self.CHI2_COST_FUNCTION(errors_to_use='pointwise')
            (self._data, self._model, self._pointwise_errors, self._par_vals, self._par_constraints)) < self._tol)

    def test_chi2_cov_mat(self):
        self.assertTrue(np.abs(
            self._cost_chi2_cov_mat - self.CHI2_COST_FUNCTION(errors_to_use='covariance')
            (self._data, self._model, self._cov_mat_inv, None, None)) < self._tol)
        self.assertTrue(np.abs(
            self._cost_chi2_cov_mat + self._par_cost - self.CHI2_COST_FUNCTION(errors_to_use='covariance')
            (self._data, self._model, self._cov_mat_inv, self._par_vals, self._par_constraints)) < self._tol)

    def test_nll_gaussian(self):
        self.assertTrue(np.abs(
            self._cost_nll_gaussian - self.NLL_COST_FUNCTION(data_point_distribution='gaussian')
            (self._data, self._model, self._pointwise_errors, None, None)) < self._tol)
        self.assertTrue(np.abs(
            self._cost_nll_gaussian + self._par_cost - self.NLL_COST_FUNCTION(data_point_distribution='gaussian')
            (self._data, self._model, self._pointwise_errors, self._par_vals, self._par_constraints)) < self._tol)

    def test_nll_poisson(self):
        self.assertTrue(np.abs(
            self._cost_nll_poisson - self.NLL_COST_FUNCTION(data_point_distribution='poisson')
            (self._data_poisson, self._model_poisson, None, None)) < self._tol)
        self.assertTrue(np.abs(
            self._cost_nll_poisson + self._par_cost - self.NLL_COST_FUNCTION(data_point_distribution='poisson')
            (self._data_poisson, self._model_poisson, self._par_vals, self._par_constraints)) < self._tol)

    def test_nllr_gaussian(self):
        self.assertTrue(np.abs(
            self._cost_chi2_pointwise - self.NLLR_COST_FUNCTION(data_point_distribution='gaussian')
            (self._data, self._model, self._pointwise_errors, None, None)) < self._tol)
        self.assertTrue(np.abs(
            self._cost_chi2_pointwise + self._par_cost - self.NLLR_COST_FUNCTION(data_point_distribution='gaussian')
            (self._data, self._model, self._pointwise_errors, self._par_vals, self._par_constraints)) < self._tol)

    def test_nllr_poisson(self):
        self.assertTrue(np.abs(
            self._cost_nllr_poisson - self.NLLR_COST_FUNCTION(data_point_distribution='poisson')
            (self._data_poisson, self._model_poisson, None, None)) < self._tol)
        self.assertTrue(np.abs(
            self._cost_nllr_poisson + self._par_cost - self.NLLR_COST_FUNCTION(data_point_distribution='poisson')
            (self._data_poisson, self._model_poisson, self._par_vals, self._par_constraints)) < self._tol)


class TestCostHist(TestCostBase):

    CHI2_COST_FUNCTION = HistCostFunction_Chi2
    NLL_COST_FUNCTION = HistCostFunction_NegLogLikelihood
    NLLR_COST_FUNCTION = HistCostFunction_NegLogLikelihoodRatio


class TestCostIndexed(TestCostBase):

    CHI2_COST_FUNCTION = IndexedCostFunction_Chi2
    NLL_COST_FUNCTION = IndexedCostFunction_NegLogLikelihood
    NLLR_COST_FUNCTION = IndexedCostFunction_NegLogLikelihoodRatio


class TestCostXY(TestCostBase):

    CHI2_COST_FUNCTION = XYCostFunction_Chi2
    NLL_COST_FUNCTION = XYCostFunction_NegLogLikelihood
    NLLR_COST_FUNCTION = XYCostFunction_NegLogLikelihoodRatio
