import unittest
import numpy as np
import six

from kafe.config import kc
from kafe.fit import XYFit
from kafe.fit.xy.fit import XYFitException
from kafe.fit.xy.model import XYModelFunctionException

CONFIG_PARAMETER_DEFAULT_VALUE = kc('core', 'default_initial_parameter_value')


class TestFittersXY(unittest.TestCase):

    @staticmethod
    def xy_model(x, a=1.1, b=2.2, c=3.3):
        return a * x ** 2 + b * x + c


    @staticmethod
    def xy_model_nodefaults(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def xy_model_partialdefaults(x, a, b, c=3.3):
        return a * x ** 2 + b * x + c

    @staticmethod
    def simple_chi2_explicit_model_name(y_data, xy_model):
        return np.sum((y_data - xy_model)**2)

    @staticmethod
    def simple_chi2(y_data, y_model):
        return np.sum((y_data - y_model)**2)

    @staticmethod
    def xy_model_reserved_names(x, a=1.1, b=2.2, c=3.3, y_data=-9.99):
        return TestFittersXY.xy_model(x, a, b, c)

    @staticmethod
    def xy_model_varargs(x, a=1.1, b=2.2, c=3.3, *args):
        return TestFittersXY.xy_model(x, a, b, c)

    @staticmethod
    def xy_model_varkwargs(x, a=1.1, b=2.2, c=3.3, **kwargs):
        return TestFittersXY.xy_model(x, a, b, c)

    @staticmethod
    def xy_model_varargs_and_varkwargs(x, a=1.1, b=2.2, c=3.3, *args, **kwargs):
        return TestFittersXY.xy_model(x, a, b, c)

    def setUp(self):
        self._ref_parameter_values = 1.1, 2.2, 3.3
        self._ref_parameter_values_partialdefault = (CONFIG_PARAMETER_DEFAULT_VALUE, CONFIG_PARAMETER_DEFAULT_VALUE, 3.3)
        self._ref_parameter_values_default = (CONFIG_PARAMETER_DEFAULT_VALUE, CONFIG_PARAMETER_DEFAULT_VALUE, CONFIG_PARAMETER_DEFAULT_VALUE)

        self._ref_x = np.arange(10)
        self._ref_y_model_values = self.xy_model(self._ref_x, *self._ref_parameter_values)
        self._ref_data_jitter = np.array([-0.3193475 , -1.2404198 , -1.4906926 , -0.78832446,
                                          -1.7638106,   0.36664261,  0.49433821,  0.0719646,
                                           1.95670326,  0.31200215])

        self._ref_y_data = self._ref_y_model_values + self._ref_data_jitter
        self._ref_xy_data = np.array([self._ref_x, self._ref_y_data])

        self.xy_fit = XYFit(xy_data=self._ref_xy_data,
                            model_function=self.xy_model,
                            cost_function=self.simple_chi2)
        self.xy_fit_explicit_model_name_in_chi2 = XYFit(
            xy_data=self._ref_xy_data,
            model_function=self.xy_model,
            cost_function=self.simple_chi2_explicit_model_name)
        self.xy_fit_default_cost_function = XYFit(xy_data=self._ref_xy_data,
                                                  model_function=self.xy_model)

        self._ref_parameter_value_estimates = [1.1351433845831516, 2.137441531781195, 2.3405503488535118]
        self._ref_y_model_value_estimates = self.xy_model(self._ref_x, *self._ref_parameter_value_estimates)


    def test_before_fit_compare_parameter_values(self):
        self.assertTrue(
            np.allclose(
                self.xy_fit.parameter_values,
                self._ref_parameter_values,
                rtol=1e-3
            )
        )

    def test_before_fit_compare_model_values(self):
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_model,
                self._ref_y_model_values,
                rtol=1e-2
            )
        )

    def test_do_fit_compare_parameter_values(self):
        self.xy_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.xy_fit.parameter_values,
                self._ref_parameter_value_estimates,
                rtol=1e-3
            )
        )

    def test_do_fit_compare_model_values(self):
        self.xy_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_model,
                self._ref_y_model_value_estimates,
                rtol=1e-2
            )
        )

    #TODO this test case seems redundant. Should it be removed?
    def test_do_fit_explicit_model_name_in_chi2_compare_parameter_values(self):
        self.xy_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.xy_fit.parameter_values,
                self._ref_parameter_value_estimates,
                rtol=1e-3
            )
        )

    def test_compare_do_fit_explicit_model_name_in_chi2(self):
        self.xy_fit.do_fit()
        self.xy_fit_explicit_model_name_in_chi2.do_fit()
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_model,
                self.xy_fit_explicit_model_name_in_chi2.y_model,
                rtol=1e-3
            )
        )
        self.assertTrue(
            np.allclose(
                self.xy_fit.parameter_values,
                self.xy_fit_explicit_model_name_in_chi2.parameter_values,
                rtol=1e-3
            )
        )

    def test_compare_do_fit_default_cost_function(self):
        self.xy_fit.do_fit()
        self.xy_fit_default_cost_function.do_fit()
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_model,
                self.xy_fit_default_cost_function.y_model,
                rtol=1e-3
            )
        )
        self.assertTrue(
            np.allclose(
                self.xy_fit.parameter_values,
                self.xy_fit_default_cost_function.parameter_values,
                rtol=1e-3
            )
        )

    def test_update_cost_function_on_parameter_change(self):
        self.xy_fit.set_all_parameter_values(self._ref_parameter_value_estimates)
        self.assertEqual(
            self.xy_fit.cost_function_value,
            self.xy_fit._cost_function(self._ref_y_data, self._ref_y_model_value_estimates),
        )

    def test_model_nodefaults(self):
        xy_fit = XYFit(xy_data=self._ref_xy_data,
                             model_function=self.xy_model_nodefaults,
                             cost_function=self.simple_chi2)
        self.assertTrue(
            np.allclose(
                xy_fit.parameter_values,
                self._ref_parameter_values_default,
                rtol=1e-3
            )
        )

    def test_model_partialdefaults(self):
        xy_fit = XYFit(xy_data=self._ref_xy_data,
                             model_function=self.xy_model_partialdefaults,
                             cost_function=self.simple_chi2)
        self.assertTrue(
            np.allclose(
                xy_fit.parameter_values,
                self._ref_parameter_values_partialdefault,
                rtol=1e-3
            )
        )

    def test_raise_reserved_parameter_names_in_model(self):
        with self.assertRaises(XYFitException):
            xy_fit_reserved_names = XYFit(
                            xy_data=self._ref_xy_data,
                            model_function=self.xy_model_reserved_names,
                            cost_function=self.simple_chi2)

    def test_raise_varargs_in_model(self):
        with self.assertRaises(XYModelFunctionException):
            xy_fit_reserved_names = XYFit(
                            xy_data=self._ref_xy_data,
                            model_function=self.xy_model_varargs,
                            cost_function=self.simple_chi2)

    def test_raise_varkwargs_in_model(self):
        with self.assertRaises(XYModelFunctionException):
            xy_fit_reserved_names = XYFit(
                            xy_data=self._ref_xy_data,
                            model_function=self.xy_model_varkwargs,
                            cost_function=self.simple_chi2)

    def test_raise_varargs_and_varkwargs_in_model(self):
        with self.assertRaises(XYModelFunctionException):
            xy_fit_reserved_names = XYFit(
                            xy_data=self._ref_xy_data,
                            model_function=self.xy_model_varargs_and_varkwargs,
                            cost_function=self.simple_chi2)

class TestFittersXYChi2WithError(unittest.TestCase):

    @staticmethod
    def xy_model(x, a=1.1, b=2.2, c=3.3):
        return a * x ** 2 + b * x + c

    @staticmethod
    def simple_chi2(y_data, y_model):
        return np.sum((y_data - y_model)**2)

    @staticmethod
    def chi2_with_error(y_data, y_model, y_data_error):
        return np.sum(((y_data - y_model)/y_data_error)**2)

    @staticmethod
    def chi2_with_cov_mat(y_data, y_model, y_total_cov_mat_inverse):
        _p = y_data - y_model
        if y_total_cov_mat_inverse is not None:
            return (_p.dot(y_total_cov_mat_inverse).dot(_p))[0, 0]
        else:
            return 9999


    def setUp(self):
        self._ref_parameter_values = 1.1, 2.2, 3.3
        self._ref_x = np.arange(10)

        self._ref_y_model_values = self.xy_model(self._ref_x, *self._ref_parameter_values)
        self._ref_data_jitter = np.array([-0.3193475 , -1.2404198 , -1.4906926 , -0.78832446,
                                          -1.7638106,   0.36664261,  0.49433821,  0.0719646,
                                           1.95670326,  0.31200215])

        self._ref_y_data = self._ref_y_model_values + self._ref_data_jitter
        self._ref_xy_data = np.array([self._ref_x, self._ref_y_data])
        self._ref_y_data_error = np.ones_like(self._ref_y_data) * 1.0
        self._ref_y_model_error = 1.0

        self.xy_fit = XYFit(xy_data=self._ref_xy_data,
                                  model_function=self.xy_model,
                                  cost_function=self.chi2_with_error)

        self.xy_fit.add_simple_error('y', self._ref_y_data_error,
                                     name="MyYDataError", correlation=0, relative=False, reference='data')
        self.xy_fit.add_simple_error('y', self._ref_y_model_error,
                                     name="MyYModelError", correlation=0, relative=False, reference='model')

        self._ref_y_total_error = np.sqrt(self._ref_y_data_error ** 2 + self._ref_y_model_error ** 2)

        self._ref_parameter_value_estimates = [1.1351433845831516, 2.137441531781195, 2.3405503488535118]
        self._ref_model_value_estimates = self.xy_model(self._ref_x, *self._ref_parameter_value_estimates)

    def test_get_matching_error_all_empty_dict(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict())
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.xy_fit._data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])
        self.assertIs(self.xy_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_get_matching_error_all_None(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=None)
        self.assertEqual(len(_errs), 2)
        print(_errs)
        self.assertIs(self.xy_fit._data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])
        self.assertIs(self.xy_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_get_matching_error_name(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict(name='MyYDataError'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.xy_fit._data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])

    def test_get_matching_error_type_simple(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict(type='simple'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.xy_fit._data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])
        self.assertIs(self.xy_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_get_matching_error_type_matrix(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict(type='matrix'))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_uncorrelated(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict(correlated=True))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_correlated(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict(correlated=False))
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.xy_fit._data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])
        self.assertIs(self.xy_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_get_matching_error_axis(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict(axis=1))
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.xy_fit._data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])
        self.assertIs(self.xy_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_get_matching_error_reference_data(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict(reference='data'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.xy_fit._data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])

    def test_get_matching_error_reference_model(self):
        _errs = self.xy_fit.get_matching_errors(matching_criteria=dict(reference='model'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.xy_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_compare_fit_chi2_errors_chi2_cov_mat(self):
        self.xy_fit.do_fit()
        self.xy_fit_chi2_with_cov_mat = XYFit(
                                  xy_data=self._ref_xy_data,
                                  model_function=self.xy_model,
                                  cost_function=self.chi2_with_cov_mat)
        self.xy_fit_chi2_with_cov_mat.add_simple_error('y', self._ref_y_data_error, correlation=0, relative=False, reference='data')
        self.xy_fit_chi2_with_cov_mat.do_fit()

        self.assertTrue(
            np.allclose(
                self.xy_fit.parameter_values,
                self.xy_fit_chi2_with_cov_mat.parameter_values
            )
        )

    def test_before_fit_compare_data_error_nexus_data_error(self):
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_data_error,
                self.xy_fit._nexus.get_values('y_data_error'),
                rtol=1e-3
            )
        )

    def test_before_fit_compare_data_cov_mat_nexus_data_cov_mat(self):
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_data_cov_mat,
                self.xy_fit._nexus.get_values('y_data_cov_mat'),
                rtol=1e-3
            )
        )

    def test_before_fit_compare_data_cov_mat_inverse_nexus_data_cov_mat_inverse(self):
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_data_cov_mat_inverse,
                self.xy_fit._nexus.get_values('y_data_cov_mat_inverse'),
                rtol=1e-3
            )
        )

    def test_before_fit_compare_parameter_values(self):
        self.assertTrue(
            np.allclose(
                self.xy_fit.parameter_values,
                self._ref_parameter_values,
                rtol=1e-3
            )
        )

    def test_before_fit_compare_model_values(self):
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_model,
                self._ref_y_model_values,
                rtol=1e-2
            )
        )

    def test_before_fit_compare_y_total_error(self):
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_total_error,
                self._ref_y_total_error,
                rtol=1e-2
            )
        )

    def test_do_fit_compare_parameter_values(self):
        self.xy_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.xy_fit.parameter_values,
                self._ref_parameter_value_estimates,
                rtol=1e-3
            )
        )

    def test_do_fit_compare_model_values(self):
        self.xy_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_model,
                self._ref_model_value_estimates,
                rtol=1e-2
            )
        )

    def test_do_fit_compare_y_total_error(self):
        self.xy_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.xy_fit.y_total_error,
                self._ref_y_total_error,
                rtol=1e-2
            )
        )

    def test_report_before_fit(self):
        _buffer = six.StringIO()
        self.xy_fit.report(output_stream=_buffer)
        self.assertNotEquals(_buffer.getvalue(), "")

    def test_report_after_fit(self):
        _buffer = six.StringIO()
        self.xy_fit.do_fit()
        self.xy_fit.report(output_stream=_buffer)
        self.assertNotEquals(_buffer.getvalue(), "")