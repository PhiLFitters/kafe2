import unittest2 as unittest
import numpy as np

from kafe2.fit import IndexedContainer, IndexedParametricModel
from kafe2.fit._base import DataContainerException
from kafe2.fit.indexed.container import IndexedContainerException
from kafe2.fit.indexed.model import IndexedParametricModelException,\
    IndexedModelFunction
from kafe2.core.error import cov_mat_from_float_list


class TestDatastoreIndexed(unittest.TestCase):

    def setUp(self):
        self._ref_data = [3.3, 5.5, 2.2, 8.5, 10.2]
        self.idx_cont = IndexedContainer(data=self._ref_data)

        self._ref_err_abs_singlevalue = 1.2
        self._ref_err_abs_valuearray = [1.2, 1.1, 1.2, 1.2, 1.1]

        self._ref_err_corr_coeff = 0.23

        self.idx_cont.add_error(self._ref_err_abs_valuearray, name='MyError', correlation=self._ref_err_corr_coeff,
                                relative=False)

        self._ref_cov_mat = cov_mat_from_float_list(self._ref_err_abs_valuearray,
                                                    correlation=self._ref_err_corr_coeff).mat

    def test_get_matching_error_all_empty_dict(self):
        _errs = self.idx_cont.get_matching_errors(matching_criteria=dict())
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.idx_cont._error_dicts['MyError']['err'], _errs['MyError'])

    def test_get_matching_error_all_None(self):
        _errs = self.idx_cont.get_matching_errors(matching_criteria=None)
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.idx_cont._error_dicts['MyError']['err'], _errs['MyError'])

    def test_get_matching_error_name(self):
        _errs = self.idx_cont.get_matching_errors(matching_criteria=dict(name='MyError'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.idx_cont._error_dicts['MyError']['err'], _errs['MyError'])

    def test_get_matching_error_type_simple(self):
        _errs = self.idx_cont.get_matching_errors(matching_criteria=dict(type='simple'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.idx_cont._error_dicts['MyError']['err'], _errs['MyError'])

    def test_get_matching_error_type_matrix(self):
        _errs = self.idx_cont.get_matching_errors(matching_criteria=dict(type='matrix'))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_uncorrelated(self):
        _errs = self.idx_cont.get_matching_errors(matching_criteria=dict(correlated=False))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_correlated(self):
        _errs = self.idx_cont.get_matching_errors(matching_criteria=dict(correlated=True))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.idx_cont._error_dicts['MyError']['err'], _errs['MyError'])

    def test_compare_error_reference(self):
        for _err_dict in self.idx_cont._error_dicts.values():
            _err_ref_vals = _err_dict['err'].reference
            self.assertTrue(
                np.all(_err_ref_vals == self._ref_data)
            )

    def test_compare_ref_data(self):
        self.assertTrue(np.all(self.idx_cont.data == self._ref_data))

    def test_compare_ref_err(self):
        self.assertTrue(np.allclose(self.idx_cont.err, self._ref_err_abs_valuearray, atol=1e-10))

    def test_compare_ref_total_cov_mat(self):
        _err = self.idx_cont.get_total_error()
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_cov_mat, atol=1e-5))

    def test_compare_ref_total_err_for_add_err_twice(self):
        self.idx_cont.add_error(self._ref_err_abs_valuearray,
                                name='MyNewError',
                                correlation=self._ref_err_corr_coeff,
                                relative=False)
        _err = self.idx_cont.get_total_error()
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_cov_mat + self._ref_cov_mat, atol=1e-5))

    def test_compare_ref_total_err_for_add_disable_new_err(self):
        self.idx_cont.add_error(self._ref_err_abs_valuearray,
                                name='MyNewError',
                                correlation=self._ref_err_corr_coeff,
                                relative=False)
        self.idx_cont.disable_error("MyNewError")
        _err = self.idx_cont.get_total_error()
        _mat = _err.cov_mat

        self.assertTrue(np.allclose(_mat, self._ref_cov_mat, atol=1e-5))

    def test_compare_ref_total_err_for_add_disable_reenable_new_err(self):
        self.idx_cont.add_error(self._ref_err_abs_valuearray,
                                name='MyNewError',
                                correlation=self._ref_err_corr_coeff,
                                relative=False)
        self.idx_cont.disable_error("MyNewError")
        _err = self.idx_cont.get_total_error()
        self.idx_cont.enable_error("MyNewError")
        _err = self.idx_cont.get_total_error()
        _mat = _err.cov_mat

        self.assertTrue(np.allclose(_mat, self._ref_cov_mat + self._ref_cov_mat, atol=1e-5))

    def test_raise_add_same_error_name_twice(self):
        self.idx_cont.add_error(0.1,
                                name="MyNewError",
                                correlation=0, relative=False)
        with self.assertRaises(DataContainerException):
            self.idx_cont.add_error(0.1,
                                    name="MyNewError",
                                    correlation=0, relative=False)

    def test_raise_get_inexistent_error(self):
        with self.assertRaises(DataContainerException):
            self.idx_cont.get_error("MyInexistentError")



class TestDatastoreIndexedParametricModel(unittest.TestCase):
    def _ref_model_func(self, slope, intercept):
        return slope * self._ref_pm_support + intercept

    def _ref_model_func_deriv_by_pars(self, slope, intercept):
        return np.array([self._ref_pm_support, [1]*len(self._ref_pm_support)])

    def setUp(self):
        self._ref_pm_support = np.linspace(-5, 5, 11)

        self._ref_params = (1.2, 3.3)
        self._ref_data = self._ref_model_func(*self._ref_params)

        self.idx_param_model = IndexedParametricModel(
            model_func=IndexedModelFunction(self._ref_model_func), 
            model_parameters=self._ref_params)

        self._test_params = (3.4, -5.23)
        self._ref_test_data = self._ref_model_func(*self._test_params)

    def test_compare_ref_data(self):
        self.assertTrue(np.all(self.idx_param_model.data == self._ref_data))

    def test_deriv_by_par(self):
        self.assertTrue(
            np.allclose(
                self.idx_param_model.eval_model_function_derivative_by_parameters(),
                self._ref_model_func_deriv_by_pars(*self._ref_params)
            )
        )


    def test_change_parameters_test_data(self):
        self.idx_param_model.parameters = self._test_params
        self.assertTrue(np.all(self.idx_param_model.data == self._ref_test_data))

    def test_raise_set_data(self):
        with self.assertRaises(IndexedParametricModelException):
            self.idx_param_model.data = self._ref_test_data
