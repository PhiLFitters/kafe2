import unittest
import numpy as np
import six

from kafe2.config import kc
from kafe2.fit import IndexedFit
from kafe2.fit.indexed.fit import IndexedFitException
from kafe2.fit.indexed.model import IndexedModelFunctionException, IndexedParametricModelException


CONFIG_PARAMETER_DEFAULT_VALUE = kc('core', 'default_initial_parameter_value')
DEFAULT_TEST_MINIMIZER = 'scipy'


class TestFittersIndexed(unittest.TestCase):

    @staticmethod
    def xy_model(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def idx_model(a=1.1, b=2.2, c=3.3):
        return TestFittersIndexed.xy_model(np.arange(10), a, b, c)

    @staticmethod
    def idx_model_nodefaults(a, b, c):
        return TestFittersIndexed.xy_model(np.arange(10), a, b, c)

    @staticmethod
    def idx_model_partialdefaults(a, b, c=3.3):
        return TestFittersIndexed.xy_model(np.arange(10), a, b, c)

    @staticmethod
    def simple_chi2_explicit_model_name(data, idx_model):
        return np.sum((data - idx_model)**2)

    @staticmethod
    def simple_chi2(data, model):
        return np.sum((data - model)**2)

    @staticmethod
    def idx_model_reserved_names(a=1.1, b=2.2, c=3.3, data=-9.99):
        return TestFittersIndexed.xy_model(np.arange(10), a, b, c)

    @staticmethod
    def idx_model_varargs(a=1.1, b=2.2, c=3.3, *args):
        return TestFittersIndexed.xy_model(np.arange(10), a, b, c)

    @staticmethod
    def idx_model_varkwargs(a=1.1, b=2.2, c=3.3, **kwargs):
        return TestFittersIndexed.xy_model(np.arange(10), a, b, c)

    @staticmethod
    def idx_model_varargs_and_varkwargs(a=1.1, b=2.2, c=3.3, *args, **kwargs):
        return TestFittersIndexed.xy_model(np.arange(10), a, b, c)

    def setUp(self):
        self._ref_parameter_values = 1.1, 2.2, 3.3
        self._ref_parameter_values_partialdefault = (CONFIG_PARAMETER_DEFAULT_VALUE, CONFIG_PARAMETER_DEFAULT_VALUE, 3.3)
        self._ref_parameter_values_default = (CONFIG_PARAMETER_DEFAULT_VALUE, CONFIG_PARAMETER_DEFAULT_VALUE, CONFIG_PARAMETER_DEFAULT_VALUE)
        self._ref_model_values = self.idx_model(*self._ref_parameter_values)
        self._ref_data_jitter = np.array([-0.3193475 , -1.2404198 , -1.4906926 , -0.78832446,
                                          -1.7638106,   0.36664261,  0.49433821,  0.0719646,
                                           1.95670326,  0.31200215])
        self._ref_data_values = self._ref_model_values + self._ref_data_jitter

        self.idx_fit = IndexedFit(data=self._ref_data_values,
                                  model_function=self.idx_model,
                                  cost_function=self.simple_chi2,
                                  minimizer=DEFAULT_TEST_MINIMIZER)
        self.idx_fit.add_simple_error(err_val=1.0)
        self.idx_fit_explicit_model_name_in_chi2 = IndexedFit(data=self._ref_data_values,
                                                              model_function=self.idx_model,
                                                              cost_function=self.simple_chi2_explicit_model_name,
                                                              minimizer=DEFAULT_TEST_MINIMIZER)
        self.idx_fit_explicit_model_name_in_chi2.add_simple_error(err_val=1.0)
        self.idx_fit_default_cost_function = IndexedFit(data=self._ref_data_values,
                                                        model_function=self.idx_model,
                                                        minimizer=DEFAULT_TEST_MINIMIZER)
        self.idx_fit_default_cost_function.add_simple_error(err_val=1.0)

        self._ref_parameter_value_estimates = [1.1351433,  2.13736919, 2.33346549]
        self._ref_model_value_estimates = self.idx_model(*self._ref_parameter_value_estimates)


    def test_before_fit_compare_parameter_values(self):
        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                self._ref_parameter_values,
                rtol=1e-3
            )
        )

    def test_before_fit_compare_model_values(self):
        self.assertTrue(
            np.allclose(
                self.idx_fit.model,
                self._ref_model_values,
                rtol=1e-3
            )
        )

    def test_do_fit_compare_parameter_values(self):
        self.idx_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                self._ref_parameter_value_estimates,
                rtol=1e-3
            )
        )

    def test_do_fit_compare_model_values(self):
        self.idx_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.model,
                self._ref_model_value_estimates,
                rtol=1e-3
            )
        )

    def test_do_fit_explicit_model_name_in_chi2_compare_parameter_values(self):
        self.idx_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                self._ref_parameter_value_estimates,
                rtol=1e-3
            )
        )

    def test_compare_do_fit_explicit_model_name_in_chi2(self):
        self.idx_fit.do_fit()
        self.idx_fit_explicit_model_name_in_chi2.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.model,
                self.idx_fit_explicit_model_name_in_chi2.model,
                rtol=1e-3
            )
        )
        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                self.idx_fit_explicit_model_name_in_chi2.parameter_values,
                rtol=1e-3
            )
        )

    def test_compare_do_fit_default_cost_function(self):
        self.idx_fit.do_fit()
        self.idx_fit_default_cost_function.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.model,
                self.idx_fit_default_cost_function.model,
                rtol=1e-3
            )
        )
        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                self.idx_fit_default_cost_function.parameter_values,
                rtol=1e-3
            )
        )

    def test_update_cost_function_on_parameter_change(self):
        self.idx_fit.set_all_parameter_values(self._ref_parameter_value_estimates)
        self.assertEqual(
            self.idx_fit.cost_function_value,
            self.idx_fit._cost_function(self._ref_data_values, self._ref_model_value_estimates),
        )

    def test_model_nodefaults(self):
        idx_fit = IndexedFit(data=self._ref_data_values,
                             model_function=self.idx_model_nodefaults,
                             cost_function=self.simple_chi2,
                             minimizer=DEFAULT_TEST_MINIMIZER)
        self.assertTrue(
            np.allclose(
                idx_fit.parameter_values,
                self._ref_parameter_values_default,
                rtol=1e-3
            )
        )

    def test_model_partialdefaults(self):
        idx_fit = IndexedFit(data=self._ref_data_values,
                             model_function=self.idx_model_partialdefaults,
                             cost_function=self.simple_chi2,
                             minimizer=DEFAULT_TEST_MINIMIZER)
        self.assertTrue(
            np.allclose(
                idx_fit.parameter_values,
                self._ref_parameter_values_partialdefault,
                rtol=1e-3
            )
        )

    def test_raise_reserved_parameter_names_in_model(self):
        with self.assertRaises(IndexedFitException):
            idx_fit_reserved_names = IndexedFit(data=self._ref_data_values,
                                                model_function=self.idx_model_reserved_names,
                                                cost_function=self.simple_chi2,
                                                minimizer=DEFAULT_TEST_MINIMIZER)

    def test_raise_varargs_in_model(self):
        with self.assertRaises(IndexedModelFunctionException):
            idx_fit_reserved_names = IndexedFit(data=self._ref_data_values,
                                                model_function=self.idx_model_varargs,
                                                cost_function=self.simple_chi2,
                                                minimizer=DEFAULT_TEST_MINIMIZER)

    def test_raise_varkwargs_in_model(self):
        with self.assertRaises(IndexedModelFunctionException):
            idx_fit_reserved_names = IndexedFit(data=self._ref_data_values,
                                                model_function=self.idx_model_varkwargs,
                                                cost_function=self.simple_chi2,
                                                minimizer=DEFAULT_TEST_MINIMIZER)

    def test_raise_varargs_and_varkwargs_in_model(self):
        with self.assertRaises(IndexedModelFunctionException):
            idx_fit_reserved_names = IndexedFit(data=self._ref_data_values,
                                                model_function=self.idx_model_varargs_and_varkwargs,
                                                cost_function=self.simple_chi2,
                                                minimizer=DEFAULT_TEST_MINIMIZER)

    def test_nexus_update_on_data_change(self):
        new_estimates = [0, 1, 0]
        self.idx_fit.data = np.arange(10)
        self.idx_fit.add_simple_error(err_val=1.0)
        self.idx_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                new_estimates,
                atol=1e-2
            )
        )

    def test_raise_on_data_length_change(self):
        with self.assertRaises(IndexedParametricModelException):
            self.idx_fit.data = np.arange(5)


class TestFittersIndexedChi2WithError(unittest.TestCase):

    @staticmethod
    def xy_model(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def idx_model(a=1.1, b=2.2, c=3.3):
        return TestFittersIndexed.xy_model(np.arange(10), a, b, c)

    @staticmethod
    def simple_chi2(data, model):
        return np.sum((data - model)**2)

    @staticmethod
    def chi2_with_error(data, model, data_error):
        return np.sum(((data - model)/data_error)**2)

    @staticmethod
    def chi2_with_cov_mat(data, model, total_cov_mat_inverse):
        _p = data - model
        if total_cov_mat_inverse is not None:
            return (_p.dot(total_cov_mat_inverse).dot(_p))
        else:
            return 9999


    def setUp(self):
        self._ref_parameter_values = 1.1, 2.2, 3.3
        self._ref_model_values = self.idx_model(*self._ref_parameter_values)
        self._ref_data_jitter = np.array([-0.3193475 , -1.2404198 , -1.4906926 , -0.78832446,
                                          -1.7638106,   0.36664261,  0.49433821,  0.0719646,
                                           1.95670326,  0.31200215])
        self._ref_data_values = self._ref_model_values + self._ref_data_jitter
        self._ref_data_error = np.ones_like(self._ref_data_values) * 1.0
        self._ref_model_error = 0.1

        self.idx_fit = IndexedFit(data=self._ref_data_values,
                                  model_function=self.idx_model,
                                  cost_function=self.chi2_with_error,
                                  minimizer=DEFAULT_TEST_MINIMIZER)

        self.idx_fit.add_simple_error(self._ref_data_error,
                                      name="MyDataError", correlation=0, relative=False, reference='data')
        self.idx_fit.add_simple_error(self._ref_model_error,
                                      name="MyModelError", correlation=0, relative=False, reference='model')

        self._ref_total_error = np.sqrt(self._ref_data_error ** 2 + self._ref_model_error ** 2)

        self._ref_parameter_value_estimates = [1.1351433, 2.13736919, 2.33346549]
        self._ref_model_value_estimates = self.idx_model(*self._ref_parameter_value_estimates)

    def test_get_matching_error_all_empty_dict(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=dict())
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.idx_fit._data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])
        self.assertIs(self.idx_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])

    def test_get_matching_error_all_None(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=None)
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.idx_fit._data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])
        self.assertIs(self.idx_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])

    def test_get_matching_error_name(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=dict(name='MyDataError'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.idx_fit._data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])

    def test_get_matching_error_type_simple(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=dict(type='simple'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.idx_fit._data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])
        self.assertIs(self.idx_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])

    def test_get_matching_error_type_matrix(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=dict(type='matrix'))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_uncorrelated(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=dict(correlated=True))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_correlated(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=dict(correlated=False))
        self.assertEqual(len(_errs), 2)
        self.assertIs(self.idx_fit._data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])
        self.assertIs(self.idx_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])

    def test_get_matching_error_reference_data(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=dict(reference='data'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.idx_fit._data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])

    def test_get_matching_error_reference_model(self):
        _errs = self.idx_fit.get_matching_errors(matching_criteria=dict(reference='model'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(self.idx_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])

    def test_compare_fit_chi2_errors_chi2_cov_mat(self):
        self.idx_fit.do_fit()
        self.idx_fit_chi2_with_cov_mat = IndexedFit(
                                  data=self._ref_data_values,
                                  model_function=self.idx_model,
                                  cost_function=self.chi2_with_cov_mat,
                                  minimizer=DEFAULT_TEST_MINIMIZER)
        self.idx_fit_chi2_with_cov_mat.add_simple_error(self._ref_data_error, correlation=0, relative=False, reference='data')
        self.idx_fit_chi2_with_cov_mat.do_fit()

        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                self.idx_fit_chi2_with_cov_mat.parameter_values,
                rtol=1e-3
            )
        )

    def test_before_fit_compare_data_error_nexus_data_error(self):
        self.assertTrue(
            np.allclose(
                self.idx_fit.data_error,
                self.idx_fit._nexus.get_values('data_error'),
                rtol=1e-3
            )
        )

    def test_before_fit_compare_data_cov_mat_nexus_data_cov_mat(self):
        self.assertTrue(
            np.allclose(
                self.idx_fit.data_cov_mat,
                self.idx_fit._nexus.get_values('data_cov_mat'),
                rtol=1e-3
            )
        )

    def test_before_fit_compare_data_cov_mat_inverse_nexus_data_cov_mat_inverse(self):
        self.assertTrue(
            np.allclose(
                self.idx_fit.data_cov_mat_inverse,
                self.idx_fit._nexus.get_values('data_cov_mat_inverse'),
                rtol=1e-3
            )
        )

    def test_before_fit_compare_parameter_values(self):
        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                self._ref_parameter_values,
                rtol=1e-3
            )
        )

    def test_before_fit_compare_model_values(self):
        self.assertTrue(
            np.allclose(
                self.idx_fit.model,
                self._ref_model_values,
                rtol=1e-3
            )
        )

    def test_before_fit_compare_total_error(self):
        self.assertTrue(
            np.allclose(
                self.idx_fit.total_error,
                self._ref_total_error,
                rtol=1e-3
            )
        )

    def test_do_fit_compare_parameter_values(self):
        self.idx_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.parameter_values,
                self._ref_parameter_value_estimates,
                rtol=1e-3
            )
        )

    def test_do_fit_compare_model_values(self):
        self.idx_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.model,
                self._ref_model_value_estimates,
                rtol=1e-3
            )
        )

    def test_do_fit_compare_total_error(self):
        self.idx_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.idx_fit.total_error,
                self._ref_total_error,
                rtol=1e-3
            )
        )

    def test_report_before_fit(self):
        _buffer = six.StringIO()
        self.idx_fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")

    def test_report_after_fit(self):
        _buffer = six.StringIO()
        self.idx_fit.do_fit()
        self.idx_fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")
