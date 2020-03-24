import unittest2 as unittest
import numpy as np

from kafe2.fit import XYContainer, XYParametricModel
from kafe2.fit._base import DataContainerException, ModelFunctionBase
from kafe2.fit.xy.container import XYContainerException
from kafe2.fit.xy.model import XYParametricModelException
from kafe2.core.error import cov_mat_from_float_list



class TestDatastoreXY(unittest.TestCase):

    def setUp(self):
        self._ref_x_data = [0, 1, 2, 3, 4]
        self._ref_y_data = [3.3, 5.5, 2.2, 8.5, 10.2]
        self.data_xy = XYContainer(x_data=self._ref_x_data, y_data=self._ref_y_data)

        self._ref_x_err_abs_singlevalue = 0.1
        self._ref_y_err_abs_singlevalue = 1.2
        self._ref_x_err_abs_valuearray = np.array([0.1, 0.1, 0.2, 0.4, 0.5])
        self._ref_y_err_abs_valuearray = [1.2, 1.1, 1.2, 1.2, 1.1]

        self._ref_x_err_corr_coeff = 0.1
        self._ref_y_err_corr_coeff = 0.23

        self.data_xy.add_error('x', self._ref_x_err_abs_valuearray, name='MyXError', correlation=self._ref_x_err_corr_coeff,
                               relative=False)
        self.data_xy.add_error('y', self._ref_y_err_abs_valuearray, name='MyYError', correlation=self._ref_y_err_corr_coeff,
                               relative=False)

        self._ref_x_cov_mat = cov_mat_from_float_list(self._ref_x_err_abs_valuearray,
                                                      correlation=self._ref_x_err_corr_coeff).mat
        self._ref_y_cov_mat = cov_mat_from_float_list(self._ref_y_err_abs_valuearray,
                                                      correlation=self._ref_y_err_corr_coeff).mat

    def test_get_matching_error_all_empty_dict(self):
        _errs = self.data_xy.get_matching_errors(matching_criteria=dict())
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.data_xy._error_dicts['MyXError']['err'], _errs['MyXError'])
        self.assertIs(self.data_xy._error_dicts['MyYError']['err'], _errs['MyYError'])

    def test_get_matching_error_all_None(self):
        _errs = self.data_xy.get_matching_errors(matching_criteria=None)
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.data_xy._error_dicts['MyXError']['err'], _errs['MyXError'])
        self.assertIs(self.data_xy._error_dicts['MyYError']['err'], _errs['MyYError'])

    def test_get_matching_error_name(self):
        _errs = self.data_xy.get_matching_errors(matching_criteria=dict(name='MyXError'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.data_xy._error_dicts['MyXError']['err'], _errs['MyXError'])

    def test_get_matching_error_type_simple(self):
        _errs = self.data_xy.get_matching_errors(matching_criteria=dict(type='simple'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.data_xy._error_dicts['MyXError']['err'], _errs['MyXError'])
        self.assertIs(self.data_xy._error_dicts['MyYError']['err'], _errs['MyYError'])

    def test_get_matching_error_type_matrix(self):
        _errs = self.data_xy.get_matching_errors(matching_criteria=dict(type='matrix'))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_uncorrelated(self):
        _errs = self.data_xy.get_matching_errors(matching_criteria=dict(correlated=False))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_correlated(self):
        _errs = self.data_xy.get_matching_errors(matching_criteria=dict(correlated=True))
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.data_xy._error_dicts['MyXError']['err'], _errs['MyXError'])
        self.assertIs(self.data_xy._error_dicts['MyYError']['err'], _errs['MyYError'])

    def test_get_matching_error_axis(self):
        _errs = self.data_xy.get_matching_errors(matching_criteria=dict(axis=1))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.data_xy._error_dicts['MyYError']['err'], _errs['MyYError'])

    def test_compare_error_reference(self):
        for _err_dict in self.data_xy._error_dicts.values():
            _err_ref_vals = _err_dict['err'].reference
            _axis = _err_dict['axis']
            assert _axis in (0, 1)
            if _axis == 0:
                self.assertTrue(
                    np.all(_err_ref_vals == self._ref_x_data)
                )
            elif _axis == 1:
                self.assertTrue(
                    np.all(_err_ref_vals == self._ref_y_data)
                )


    def test_compare_ref_x_data(self):
        self.assertTrue(np.all(self.data_xy.x == self._ref_x_data))

    def test_compare_ref_y_data(self):
        self.assertTrue(np.all(self.data_xy.y == self._ref_y_data))

    def test_compare_ref_x_err(self):
        self.assertTrue(np.allclose(self.data_xy.x_err, self._ref_x_err_abs_valuearray, atol=1e-10))

    def test_compare_ref_y_err(self):
        self.assertTrue(np.allclose(self.data_xy.y_err, self._ref_y_err_abs_valuearray, atol=1e-10))

    # def test_compare_ref_x_cov_mat(self):
    #     self.assertTrue(np.allclose(self.data_xy.???, self._ref_x_cov_mat, atol=1e-5))
    #
    # def test_compare_ref_y_cov_mat(self):
    #     self.assertTrue(np.allclose(self.data_xy.???, self._ref_y_cov_mat, atol=1e-5))

    def test_compare_ref_total_x_cov_mat(self):
        _err = self.data_xy.get_total_error(0)
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_x_cov_mat, atol=1e-5))

    def test_compare_ref_total_y_cov_mat(self):
        _err = self.data_xy.get_total_error(1)
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_y_cov_mat, atol=1e-5))

    def test_compare_ref_total_y_for_x_plus_y_as_y_err(self):
        self.data_xy.add_error('y', self._ref_x_err_abs_valuearray,
                               correlation=self._ref_x_err_corr_coeff,
                               relative=False)
        _err = self.data_xy.get_total_error(1)
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_y_cov_mat + self._ref_x_cov_mat, atol=1e-5))

    def test_compare_ref_total_y_for_x_plus_y_as_y_err_disabled(self):
        self.data_xy.add_error('y', self._ref_x_err_abs_valuearray,
                               name="MyNewYError",
                               correlation=self._ref_x_err_corr_coeff,
                               relative=False)
        self.data_xy.disable_error("MyNewYError")
        _err = self.data_xy.get_total_error(1)
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_y_cov_mat, atol=1e-5))

    def test_compare_ref_total_y_for_x_plus_y_as_y_err_disabled_reenabled(self):
        self.data_xy.add_error('y', self._ref_x_err_abs_valuearray,
                               name="MyNewYError",
                               correlation=self._ref_x_err_corr_coeff,
                               relative=False)
        self.data_xy.disable_error("MyNewYError")
        _err = self.data_xy.get_total_error(1)
        self.data_xy.enable_error("MyNewYError")
        _err = self.data_xy.get_total_error(1)
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_y_cov_mat + self._ref_x_cov_mat, atol=1e-5))

    def test_raise_add_same_error_name_twice(self):
        self.data_xy.add_error('y', 0.1,
                               name="MyNewYError",
                               correlation=0, relative=False)
        with self.assertRaises(DataContainerException):
            self.data_xy.add_error('y', 0.1,
                                   name="MyNewYError",
                                   correlation=0, relative=False)

    def test_raise_get_inexistent_error(self):
        with self.assertRaises(DataContainerException):
            self.data_xy.get_error("MyInexistentYError")


class TestDatastoreXYParametricModel(unittest.TestCase):

    @staticmethod
    def _ref_model_func(x, slope, intercept):
        return slope * x + intercept

    def _ref_model_func_deriv_by_x(x, slope, intercept):
        return slope

    _ref_model_func_deriv_by_x = np.vectorize(_ref_model_func_deriv_by_x)
    _ref_model_func_deriv_by_x = staticmethod(_ref_model_func_deriv_by_x)

    def _ref_model_func_deriv_by_pars(x, slope, intercept):
        return [x, 1]

    _ref_model_func_deriv_by_pars = np.vectorize(_ref_model_func_deriv_by_pars)
    _ref_model_func_deriv_by_pars = staticmethod(_ref_model_func_deriv_by_pars)

    def setUp(self):
        self._ref_x = np.linspace(-5, 5, 11)
        self._test_x = np.linspace(-3, 45, 20)

        self._ref_params = (1.2, 3.3)
        self._ref_data = self._ref_model_func(self._ref_x, *self._ref_params)

        self.xy_param_model = XYParametricModel(
            x_data=self._ref_x, 
            model_func=ModelFunctionBase(self._ref_model_func),
            model_parameters=self._ref_params)

        self._test_params = (3.4, -5.23)
        self._ref_data_ref_x_test_params =  self._ref_model_func(self._ref_x, *self._test_params)
        self._ref_data_test_x_ref_params =  self._ref_model_func(self._test_x, *self._ref_params)
        self._ref_data_test_x_test_params = self._ref_model_func(self._test_x, *self._test_params)



    def test_compare_ref_data(self):
        self.assertTrue(np.all(self.xy_param_model.y == self._ref_data))

    def test_deriv_by_x(self):
        self.assertTrue(
            np.allclose(
                self.xy_param_model.eval_model_function_derivative_by_x(),
                self._ref_model_func_deriv_by_x(self._ref_data, *self._ref_params)
            )
        )

    def test_deriv_by_par(self):
        _dp = [self._ref_model_func_deriv_by_pars(x, *self._ref_params) for x in self._ref_x]
        _dp = np.array(_dp).T
        self.assertTrue(
            np.allclose(
                self.xy_param_model.eval_model_function_derivative_by_parameters(),
                _dp
            )
        )


    def test_change_parameters_test_data(self):
        self.xy_param_model.parameters = self._test_params
        self.assertTrue(np.allclose(self.xy_param_model.y, self._ref_data_ref_x_test_params))

    def test_change_x_test_data(self):
        self.xy_param_model.x = self._test_x
        self.assertTrue(np.allclose(self.xy_param_model.y, self._ref_data_test_x_ref_params))

    def test_change_x_change_parameters_test_data(self):
        self.xy_param_model.x = self._test_x
        self.xy_param_model.parameters = self._test_params
        self.assertTrue(np.allclose(self.xy_param_model.y, self._ref_data_test_x_test_params))

    def test_change_parameters_change_x_test_data(self):
        self.xy_param_model.parameters = self._test_params
        self.xy_param_model.x = self._test_x
        self.assertTrue(np.allclose(self.xy_param_model.y, self._ref_data_test_x_test_params))

    def test_raise_set_data(self):
        with self.assertRaises(XYParametricModelException):
            self.xy_param_model.data = self._ref_data_ref_x_test_params

    def test_raise_set_y(self):
        with self.assertRaises(XYParametricModelException):
            self.xy_param_model.y = self._ref_data_ref_x_test_params
