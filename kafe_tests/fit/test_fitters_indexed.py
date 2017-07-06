import unittest
import numpy as np

from kafe.config import kc
from kafe.fit import IndexedFit
from kafe.fit.indexed.fit import IndexedFitException
from kafe.fit.indexed.model import IndexedModelFunctionException


CONFIG_PARAMETER_DEFAULT_VALUE = kc('core', 'default_initial_parameter_value')


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
                                  cost_function=self.simple_chi2)
        self.idx_fit_explicit_model_name_in_chi2 = IndexedFit(
            data=self._ref_data_values,
            model_function=self.idx_model,
            cost_function=self.simple_chi2_explicit_model_name)
        self.idx_fit_default_cost_function = IndexedFit(data=self._ref_data_values,
                                                        model_function=self.idx_model)

        self._ref_parameter_value_estimates = [1.1351433845831516, 2.137441531781195, 2.3405503488535118]
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
                rtol=1e-2
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
                rtol=1e-2
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
                             cost_function=self.simple_chi2)
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
                             cost_function=self.simple_chi2)
        self.assertTrue(
            np.allclose(
                idx_fit.parameter_values,
                self._ref_parameter_values_partialdefault,
                rtol=1e-3
            )
        )

    def test_raise_reserved_parameter_names_in_model(self):
        with self.assertRaises(IndexedFitException):
            idx_fit_reserved_names = IndexedFit(
                            data=self._ref_data_values,
                            model_function=self.idx_model_reserved_names,
                            cost_function=self.simple_chi2)

    def test_raise_varargs_in_model(self):
        with self.assertRaises(IndexedModelFunctionException):
            idx_fit_reserved_names = IndexedFit(
                            data=self._ref_data_values,
                            model_function=self.idx_model_varargs,
                            cost_function=self.simple_chi2)

    def test_raise_varkwargs_in_model(self):
        with self.assertRaises(IndexedModelFunctionException):
            idx_fit_reserved_names = IndexedFit(
                            data=self._ref_data_values,
                            model_function=self.idx_model_varkwargs,
                            cost_function=self.simple_chi2)

    def test_raise_varargs_and_varkwargs_in_model(self):
        with self.assertRaises(IndexedModelFunctionException):
            idx_fit_reserved_names = IndexedFit(
                            data=self._ref_data_values,
                            model_function=self.idx_model_varargs_and_varkwargs,
                            cost_function=self.simple_chi2)

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
            return (_p.dot(total_cov_mat_inverse).dot(_p))[0, 0]
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

        self.idx_fit = IndexedFit(data=self._ref_data_values,
                                  model_function=self.idx_model,
                                  cost_function=self.chi2_with_error)

        self.idx_fit.add_simple_error(self._ref_data_error, correlation=0, relative=False)

        self._ref_parameter_value_estimates = [1.1351433845831516, 2.137441531781195, 2.3405503488535118]
        self._ref_model_value_estimates = self.idx_model(*self._ref_parameter_value_estimates)

    def test_compare_fit_chi2_errors_chi2_cov_mat(self):
        self.idx_fit.do_fit()
        self.idx_fit_chi2_with_cov_mat = IndexedFit(
                                  data=self._ref_data_values,
                                  model_function=self.idx_model,
                                  cost_function=self.chi2_with_cov_mat)
        self.idx_fit_chi2_with_cov_mat.add_simple_error(self._ref_data_error, correlation=0, relative=False)
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
                rtol=1e-2
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
                rtol=1e-2
            )
        )