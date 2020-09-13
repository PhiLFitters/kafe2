import numpy as np
from scipy.stats import norm
import unittest2 as unittest
import six
from abc import ABCMeta

from kafe2 import HistContainer, HistFit, IndexedFit, MultiFit, XYFit
from kafe2.test.tools import calculate_expected_fit_parameters_xy
from kafe2.test.fit.test_fit import AbstractTestFit
from kafe2.fit.util.function_library import quadratic_model, quadratic_model_derivative

_cannot_import_IMinuit = False
try:
    from kafe2.core.minimizers.iminuit_minimizer import MinimizerIMinuit
except (ImportError, SyntaxError):
    _cannot_import_IMinuit = True


@six.add_metaclass(ABCMeta)
class TestMultiFit(AbstractTestFit, unittest.TestCase):

    @staticmethod
    def _split_data(data, axis=0):
        _random = np.random.rand(data.shape[axis])
        _split_indices_1 = np.argwhere(_random < 0.5).flatten()
        _split_indices_2 = np.argwhere(_random >= 0.5).flatten()
        if axis == 0:
            return data[_split_indices_1], data[_split_indices_2]
        if axis == 1:
            return data[:, _split_indices_1], data[:, _split_indices_2]
        raise Exception()

    def _assert_fits_valid_and_equal(
            self, _fit_1, _fit_2, atol=5e-3, rtol=2e-4, check_cost_function_value=True):
        with self.subTest("parameter_names"):
            assert len(_fit_1.parameter_names) == len(_fit_2.parameter_names)
            for _par_name_1, _par_name_2 in zip(_fit_1.parameter_names, _fit_2.parameter_names):
                assert _par_name_1 == _par_name_2
        self._assert_fit_results_equal(
            _fit_1, _fit_2, rtol=rtol, atol=atol,
            check_cost_function_value=check_cost_function_value)

        _fit_1.do_fit()
        _fit_2.do_fit()

        with self.subTest("parameter_values changed during fit"):
            for _parameter_value in _fit_1.parameter_values:
                assert _parameter_value != 1.0
            for _parameter_value in _fit_2.parameter_values:
                assert _parameter_value != 1.0
        self._assert_fit_results_equal(
            _fit_1, _fit_2, rtol=rtol, atol=atol,
            check_cost_function_value=check_cost_function_value)

    def _assert_fits_valid_and_equal_double(
            self, regular_fit, double_fit, atol=5e-3, rtol=2e-4, check_cost_function_value=False):
        with self.subTest("parameter_names"):
            assert len(regular_fit.parameter_names) == len(double_fit.parameter_names)
            for _par_name_1, _par_name_2 in zip(regular_fit.parameter_names, double_fit.parameter_names):
                assert _par_name_1 == _par_name_2
        self._assert_fit_results_equal(
            regular_fit, double_fit, rtol=rtol, atol=atol,
            check_cost_function_value=check_cost_function_value, fit_2_is_double_fit=True)

        regular_fit.do_fit()
        double_fit.do_fit()

        with self.subTest("parameter_values changed during fit"):
            for _parameter_value in regular_fit.parameter_values:
                assert _parameter_value != 1.0
            for _parameter_value in double_fit.parameter_values:
                assert _parameter_value != 1.0
        self._assert_fit_results_equal(
            regular_fit, double_fit, rtol=rtol, atol=atol,
            check_cost_function_value=check_cost_function_value, fit_2_is_double_fit=True)

    @staticmethod
    def _get_hist_data(loc=2.5, scale=0.5):
        return np.random.normal(loc=loc, scale=scale, size=1000)

    @staticmethod
    def _norm_cdf(x, mu, sigma):
        return norm.cdf(x=x, loc=mu, scale=sigma)

    def _get_hist_fit(self, fill_data):
        return HistFit(
            HistContainer(
                n_bins=100,
                bin_range=(1.5, 3.5),
                fill_data=fill_data
            ),
            bin_evaluation=TestMultiFit._norm_cdf,
            minimizer=self._minimizer
        )

    def _set_hist_fits(self):
        _raw_data = TestMultiFit._get_hist_data()
        self._fit_hist_all = self._get_hist_fit(_raw_data)
        _split_data_1, _split_data_2 = self._split_data(_raw_data)
        self._fit_hist_all_multi = MultiFit([self._get_hist_fit(_raw_data)])
        self._fit_hist_split_multi = MultiFit(fit_list=[
            self._get_hist_fit(_split_data_1),
            self._get_hist_fit(_split_data_2),
        ])
        self._fit_hist_split_1 = self._get_hist_fit(_split_data_1)
        self._fit_hist_split_1_multi = MultiFit(
            [self._get_hist_fit(_split_data_1)])
        self._fit_hist_split_2 = self._get_hist_fit(_split_data_2)
        self._fit_hist_split_2_multi = MultiFit(
            [self._get_hist_fit(_split_data_2)])

    @staticmethod
    def _get_indexed_data(err_y=0.01, a=1.2, b=2.3, c=3.4):
        _x_0 = np.linspace(start=-10, stop=10, num=101, endpoint=True)
        _y_0 = quadratic_model(_x_0, a, b, c)
        _y_jitter = np.random.normal(loc=0, scale=err_y, size=101)
        _y_data = _y_0 + _y_jitter

        return _y_data

    @staticmethod
    def quadratic_model_indexed_all(a, b, c):
        _x = np.linspace(start=-10, stop=10, num=101, endpoint=True)
        return quadratic_model(_x, a, b, c)

    @staticmethod
    def quadratic_model_indexed_split_1(a, b, c):
        _x = np.linspace(start=-10, stop=0, num=50, endpoint=False)
        return quadratic_model(_x, a, b, c)

    @staticmethod
    def quadratic_model_indexed_split_2(a, b, c):
        _x = np.linspace(start=0, stop=10, num=51, endpoint=True)
        return quadratic_model(_x, a, b, c)

    def _get_indexed_fit(self, data, model_function):
        _err = dict(err_val=0.1, relative=True, reference="model") \
            if self._relative_model_error else dict(err_val=0.1)
        _indexed_fit = IndexedFit(
            data=data,
            model_function=model_function,
            cost_function='chi2',
            minimizer=self._minimizer
        )
        _indexed_fit.add_error(**_err)
        return _indexed_fit

    def _set_indexed_fits(self):
        _data = TestMultiFit._get_indexed_data()
        self._fit_indexed_all = self._get_indexed_fit(
            _data, TestMultiFit.quadratic_model_indexed_all)
        _split_data_1 = _data[:50]
        _split_data_2 = _data[50:]
        self._fit_indexed_all_multi = MultiFit([self._get_indexed_fit(
            _data, TestMultiFit.quadratic_model_indexed_all)])
        self._fit_indexed_split_multi = MultiFit(fit_list=[
            self._get_indexed_fit(
                _split_data_1, TestMultiFit.quadratic_model_indexed_split_1),
            self._get_indexed_fit(
                _split_data_2, TestMultiFit.quadratic_model_indexed_split_2)
        ])
        self._fit_indexed_split_1 = self._get_indexed_fit(
            _split_data_1, TestMultiFit.quadratic_model_indexed_split_1)
        self._fit_indexed_split_1_multi = MultiFit([self._get_indexed_fit(
            _split_data_1, TestMultiFit.quadratic_model_indexed_split_1)])
        self._fit_indexed_split_2 = self._get_indexed_fit(
            _split_data_2, TestMultiFit.quadratic_model_indexed_split_2)
        self._fit_indexed_split_2_multi = MultiFit([self._get_indexed_fit(
            _split_data_2, TestMultiFit.quadratic_model_indexed_split_2)])

    @staticmethod
    def _get_xy_data(err_x=0.01, err_y=0.01, a=1.2, b=2.3, c=3.4):
        _x_0 = np.linspace(start=-10, stop=10, num=25, endpoint=True)
        _x_jitter = np.random.normal(loc=0, scale=err_x, size=25)
        _x_data = _x_0 + _x_jitter

        _y_0 = quadratic_model(_x_data, a, b, c)
        _y_jitter = np.random.normal(loc=0, scale=err_y, size=25)
        _y_data = _y_0 + _y_jitter

        return np.array([_x_data, _y_data])

    def _get_xy_fit(self, xy_data):
        _err_x = dict(axis="x", err_val=0.01, name="x") if self._x_error is not None else None
        _err_y = dict(axis="y", err_val=0.1, relative=True, reference="model", name="y") \
            if self._relative_model_error else dict(axis="y", err_val=0.01, name="y")
        if self._x_error is None:
            _xy_fit = XYFit(
                xy_data=xy_data,
                model_function=quadratic_model,
                cost_function='chi2',
                minimizer=self._minimizer
            )
        else:
            _xy_fit = XYFit(
                xy_data=xy_data,
                model_function=quadratic_model,
                cost_function='chi2',
                minimizer=self._minimizer,
                dynamic_error_algorithm=self._x_error
            )
        if _err_x is not None:
            _xy_fit.add_error(**_err_x)
        _xy_fit.add_error(**_err_y)
        return _xy_fit

    def _set_xy_fits(self):
        self._xy_data = TestMultiFit._get_xy_data()
        self._fit_xy_all = self._get_xy_fit(self._xy_data)
        self._split_data_1, self._split_data_2 = TestMultiFit._split_data(self._xy_data, axis=1)
        self._fit_xy_all_multi = MultiFit(
            [self._get_xy_fit(self._xy_data)])
        self._fit_xy_split_multi = MultiFit(fit_list=[
            self._get_xy_fit(self._split_data_1),
            self._get_xy_fit(self._split_data_2)
        ])
        self._fit_xy_split_1 = self._get_xy_fit(self._split_data_1)
        self._fit_xy_split_1_multi = MultiFit(
            [self._get_xy_fit(self._split_data_1)])
        self._fit_xy_split_2 = self._get_xy_fit(self._split_data_2)
        self._fit_xy_split_2_multi = MultiFit(
            [self._get_xy_fit(self._split_data_2)])
        self._fit_xy_all_double = MultiFit([
            self._get_xy_fit(self._xy_data),
            self._get_xy_fit(self._xy_data)
        ])

    def _set_expected_xy_par_values(self):
        _x_err_val = 0.01 if self._x_error else 0.0
        # Note that when self._relative_model_error is True data errors are added here.
        self._expected_parameters_all = calculate_expected_fit_parameters_xy(
            x_data=self._xy_data[0], y_data=self._xy_data[1], model_function=quadratic_model,
            y_error=0.1 * self._xy_data[1] if self._relative_model_error else 0.01,
            initial_parameter_values=[1.0, 1.0, 1.0], x_error=_x_err_val,
            model_function_derivative=quadratic_model_derivative
        )
        self._expected_parameters_split_1 = calculate_expected_fit_parameters_xy(
            x_data=self._split_data_1[0], y_data=self._split_data_1[1],
            model_function=quadratic_model,
            y_error=0.1 * self._split_data_1[1] if self._relative_model_error else 0.01,
            initial_parameter_values=[1.0, 1.0, 1.0], x_error=_x_err_val,
            model_function_derivative=quadratic_model_derivative
        )
        self._expected_parameters_split_2 = calculate_expected_fit_parameters_xy(
            x_data=self._split_data_2[0], y_data=self._split_data_2[1],
            model_function=quadratic_model,
            y_error=0.1 * self._split_data_2[1] if self._relative_model_error else 0.01,
            initial_parameter_values=[1.0, 1.0, 1.0], x_error=_x_err_val,
            model_function_derivative=quadratic_model_derivative
        )

    def setUp(self):
        np.random.seed(1337)
        self._minimizer = "scipy"
        self._relative_model_error = False
        self._x_error = None


class TestMultiFitIntegrityHist(TestMultiFit):

    def setUp(self):
        TestMultiFit.setUp(self)

    def test_split_fit_integrity_simple_scipy(self):
        self._set_hist_fits()
        self._assert_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_all_multi)
        self._assert_fits_valid_and_equal(self._fit_hist_split_1, self._fit_hist_split_1_multi)
        self._assert_fits_valid_and_equal(self._fit_hist_split_2, self._fit_hist_split_2_multi)

    def test_split_fit_vs_regular_fit_scipy(self):
        self._set_hist_fits()
        self._assert_fits_valid_and_equal(
            self._fit_hist_all, self._fit_hist_split_multi, check_cost_function_value=False
        )


class TestMultiFitIntegrityIndexed(TestMultiFit):

    def setUp(self):
        TestMultiFit.setUp(self)

    def test_split_fit_integrity_simple_scipy(self):
        self._set_indexed_fits()
        self._assert_fits_valid_and_equal(self._fit_indexed_all, self._fit_indexed_all_multi)
        self._assert_fits_valid_and_equal(
            self._fit_indexed_split_1, self._fit_indexed_split_1_multi)
        self._assert_fits_valid_and_equal(
            self._fit_indexed_split_2, self._fit_indexed_split_2_multi)

    def test_split_fit_vs_regular_fit_scipy(self):
        self._set_indexed_fits()
        self._assert_fits_valid_and_equal(self._fit_indexed_all, self._fit_indexed_split_multi)

    def test_split_fit_integrity_simple_scipy_with_relative_error(self):
        self._relative_model_error = True
        self._set_indexed_fits()
        self._assert_fits_valid_and_equal(
            self._fit_indexed_all, self._fit_indexed_all_multi)
        self._assert_fits_valid_and_equal(
            self._fit_indexed_split_1, self._fit_indexed_split_1_multi)
        self._assert_fits_valid_and_equal(
            self._fit_indexed_split_2, self._fit_indexed_split_2_multi)

    def test_split_fit_vs_regular_fit_scipy_with_relative_error(self):
        self._relative_model_error = True
        self._set_indexed_fits()
        self._assert_fits_valid_and_equal(
            self._fit_indexed_all, self._fit_indexed_split_multi)


class TestMultiFitIntegrityXY(TestMultiFit):

    def setUp(self):
        TestMultiFit.setUp(self)

    def _assert_fits_match_expectation(self, atol, rtol):
        self._fit_xy_all.do_fit()
        self._assert_values_equal(
            "all", self._expected_parameters_all, self._fit_xy_all.parameter_values,
            atol=atol, rtol=rtol)
        self._fit_xy_all_multi.do_fit()
        self._assert_values_equal(
            "all", self._expected_parameters_all, self._fit_xy_all_multi.parameter_values,
            atol=atol, rtol=rtol)
        self._fit_xy_split_multi.do_fit()
        self._assert_values_equal(
            "all", self._expected_parameters_all, self._fit_xy_split_multi.parameter_values,
            atol=atol, rtol=rtol)
        self._fit_xy_split_1.do_fit()
        self._assert_values_equal(
            "all", self._expected_parameters_split_1, self._fit_xy_split_1.parameter_values,
            atol=atol, rtol=rtol)
        self._fit_xy_split_1_multi.do_fit()
        self._assert_values_equal(
            "all", self._expected_parameters_split_1, self._fit_xy_split_1_multi.parameter_values,
            atol=atol, rtol=rtol)
        self._fit_xy_split_2.do_fit()
        self._assert_values_equal(
            "all", self._expected_parameters_split_2, self._fit_xy_split_2.parameter_values,
            atol=atol, rtol=rtol)
        self._fit_xy_split_2_multi.do_fit()
        self._assert_values_equal(
            "all", self._expected_parameters_split_2, self._fit_xy_split_2_multi.parameter_values,
            atol=0, rtol=rtol)

    def test_parameter_values_match_expectation(self):
        self._set_xy_fits()
        self._set_expected_xy_par_values()
        self._assert_fits_match_expectation(atol=0, rtol=1e-6)

    def test_parameter_values_match_expectation_relative_y_error_nonlinear(self):
        self._relative_model_error = True
        self._set_xy_fits()
        self._set_expected_xy_par_values()
        self._assert_fits_match_expectation(atol=0, rtol=2e-3)

    def test_parameter_values_match_expectation_relative_y_error_iterative(self):
        self._x_error = "iterative"
        self._relative_model_error = True
        self._set_xy_fits()
        self._set_expected_xy_par_values()
        self._assert_fits_match_expectation(atol=0, rtol=1e-4)

    def test_parameter_values_match_expectation_nonlinear_x_error(self):
        self._x_error = "nonlinear"
        self._set_xy_fits()
        self._set_expected_xy_par_values()
        self._assert_fits_match_expectation(atol=0, rtol=1e-6)

    def test_parameter_values_match_expectation_iterative_linear_x_error(self):
        self._x_error = "iterative"
        self._set_xy_fits()
        self._set_expected_xy_par_values()
        self._assert_fits_match_expectation(atol=0, rtol=1e-5)

    def test_split_fit_integrity_simple(self):
        self._set_xy_fits()
        self._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_all_multi)
        self._assert_fits_valid_and_equal(self._fit_xy_split_1, self._fit_xy_split_1_multi)
        self._assert_fits_valid_and_equal(self._fit_xy_split_2, self._fit_xy_split_2_multi)

    def test_split_fit_vs_regular_fit(self):
        self._set_xy_fits()
        self._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)

    def test_split_fit_vs_regular_fit_with_nonlinear_relative_model_error(self):
        self._x_error = "nonlinear"
        self._relative_model_error = True
        self._set_xy_fits()
        self._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)

    def test_split_fit_vs_regular_fit_with_iterative_relative_model_error(self):
        self._x_error = "iterative"
        self._relative_model_error = True
        self._set_xy_fits()
        self._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)

    def test_double_fit_vs_regular_fit(self):
        self._set_xy_fits()
        self._assert_fits_valid_and_equal_double(self._fit_xy_all, self._fit_xy_all_double)

    def test_split_fit_vs_regular_fit_with_nonlinear_x_error(self):
        self._x_error = "nonlinear"
        self._set_xy_fits()
        self._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)

    def test_split_fit_vs_regular_fit_with_iterative_x_error(self):
        self._x_error = "iterative"
        self._set_xy_fits()
        self._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)

    def test_shared_error(self):
        self._set_xy_fits()
        self._fit_xy_all.add_error(axis="y", err_val=1.0)
        self._fit_xy_all_double.add_error(err_val=1.0, fits="all", axis="y")
        self._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_all_double)
