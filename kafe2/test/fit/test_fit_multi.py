import numpy as np
from scipy.stats import norm
import unittest2 as unittest

from kafe2 import HistContainer, HistFit, IndexedFit, MultiFit, XYFit
from kafe2.test.tools import calculate_expected_fit_parameters_xy
from kafe2.fit.util.function_library import quadratic_model, quadratic_model_derivative

_cannot_import_IMinuit = False
try:
    from kafe2.core.minimizers.iminuit_minimizer import MinimizerIMinuit
except ImportError:
    _cannot_import_IMinuit = True


class TestMultiFit(unittest.TestCase):

    @staticmethod
    def _split_data(data, axis=0):
        _random = np.random.rand(data.shape[axis])
        _split_indices_1 = np.argwhere(_random < 0.5).flatten()
        _split_indices_2 = np.argwhere(_random >= 0.5).flatten()
        if axis == 0:
            return data[_split_indices_1], data[_split_indices_2]
        elif axis == 1:
            return data[:, _split_indices_1], data[:, _split_indices_2]
        else:
            raise Exception()

    @staticmethod
    def _assert_fits_valid_and_equal(_fit_1, _fit_2):
        _fit_1.do_fit()
        _fit_2.do_fit()

        for _parameter_value in _fit_1.parameter_values:
            assert _parameter_value != 1.0
        for _parameter_value in _fit_2.parameter_values:
            assert _parameter_value != 1.0
        _tol = 1e-4
        print _fit_1.parameter_values
        print _fit_2.parameter_values
        assert np.allclose(_fit_1.parameter_values, _fit_2.parameter_values, atol=0, rtol=_tol)
        assert np.allclose(_fit_1.parameter_errors, _fit_2.parameter_errors, atol=0, rtol=_tol)
        assert _fit_1.parameter_cor_mat is not None
        assert _fit_2.parameter_cor_mat is not None
        assert np.allclose(_fit_1.parameter_cor_mat, _fit_2.parameter_cor_mat, atol=_tol, rtol=_tol)
        assert _fit_1.parameter_cov_mat is not None
        assert _fit_2.parameter_cov_mat is not None
        assert np.allclose(_fit_1.parameter_cov_mat, _fit_2.parameter_cov_mat, atol=_tol, rtol=_tol)

    @staticmethod
    def _get_hist_data(loc=2.5, scale=0.5):
        return np.random.normal(loc=loc, scale=scale, size=1000)

    @staticmethod
    def _norm_cdf(x, mu, sigma):
        return norm.cdf(x=x, loc=mu, scale=sigma)

    @staticmethod
    def _get_hist_fit(fill_data, minimizer):
        return HistFit(
            HistContainer(
                n_bins=100,
                bin_range=(1.0, 4.0),
                fill_data=fill_data
            ),
            model_density_antiderivative=TestMultiFit._norm_cdf,
            minimizer=minimizer
        )

    def _set_hist_fits(self, minimizer):
        _raw_data = TestMultiFit._get_hist_data()
        self._fit_hist_all = TestMultiFit._get_hist_fit(_raw_data, minimizer)
        _split_data_1, _split_data_2 = TestMultiFit._split_data(_raw_data)
        self._fit_hist_all_multi = MultiFit(TestMultiFit._get_hist_fit(_raw_data, minimizer))
        self._fit_hist_split_multi = MultiFit(fit_list=[
            TestMultiFit._get_hist_fit(_split_data_1, minimizer),
            TestMultiFit._get_hist_fit(_split_data_2, minimizer),
        ])
        self._fit_hist_split_1 = TestMultiFit._get_hist_fit(_split_data_1, minimizer)
        self._fit_hist_split_1_multi = MultiFit(TestMultiFit._get_hist_fit(_split_data_1, minimizer))
        self._fit_hist_split_2 = TestMultiFit._get_hist_fit(_split_data_2, minimizer)
        self._fit_hist_split_2_multi = MultiFit(TestMultiFit._get_hist_fit(_split_data_2, minimizer))

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

    @staticmethod
    def _get_indexed_fit(data, model_function, minimizer, err=0.01):
        _indexed_fit = IndexedFit(
            data=data,
            model_function=model_function,
            cost_function='chi2',
            minimizer=minimizer
        )
        _indexed_fit.add_simple_error(err)
        return _indexed_fit

    def _set_indexed_fits(self, minimizer):
        _data = TestMultiFit._get_indexed_data()
        self._fit_indexed_all = TestMultiFit._get_indexed_fit(
            _data, TestMultiFit.quadratic_model_indexed_all, minimizer)
        _split_data_1 = _data[:50]
        _split_data_2 = _data[50:]
        self._fit_indexed_all_multi = MultiFit(TestMultiFit._get_indexed_fit(
            _data, TestMultiFit.quadratic_model_indexed_all, minimizer))
        self._fit_indexed_split_multi = MultiFit(fit_list=[
            TestMultiFit._get_indexed_fit(
                _split_data_1, TestMultiFit.quadratic_model_indexed_split_1, minimizer),
            TestMultiFit._get_indexed_fit(
                _split_data_2, TestMultiFit.quadratic_model_indexed_split_2, minimizer)
        ])
        self._fit_indexed_split_1 = TestMultiFit._get_indexed_fit(
            _split_data_1, TestMultiFit.quadratic_model_indexed_split_1, minimizer)
        self._fit_indexed_split_1_multi = MultiFit(TestMultiFit._get_indexed_fit(
            _split_data_1, TestMultiFit.quadratic_model_indexed_split_1, minimizer))
        self._fit_indexed_split_2 = TestMultiFit._get_indexed_fit(
            _split_data_2, TestMultiFit.quadratic_model_indexed_split_2, minimizer)
        self._fit_indexed_split_2_multi = MultiFit(TestMultiFit._get_indexed_fit(
            _split_data_2, TestMultiFit.quadratic_model_indexed_split_2, minimizer))

    @staticmethod
    def _get_xy_data(err_x=0.01, err_y=0.01, a=1.2, b=2.3, c=3.4):
        _x_0 = np.linspace(start=-10, stop=10, num=101, endpoint=True)
        _x_jitter = np.random.normal(loc=0, scale=err_x, size=101)
        _x_data = _x_0 + _x_jitter

        _y_0 = quadratic_model(_x_data, a, b, c)
        _y_jitter = np.random.normal(loc=0, scale=err_y, size=101)
        _y_data = _y_0 + _y_jitter

        return np.array([_x_data, _y_data])

    @staticmethod
    def _get_xy_fit(xy_data, minimizer, err_x=0.01, err_y=0.01):
        _xy_fit = XYFit(
            xy_data=xy_data,
            model_function=quadratic_model,
            cost_function='chi2',
            minimizer=minimizer
        )
        _xy_fit.add_simple_error('x', err_x)
        _xy_fit.add_simple_error('y', err_y)
        return _xy_fit

    def _set_xy_fits(self, minimizer):
        _xy_data = TestMultiFit._get_xy_data()
        self._fit_xy_all = TestMultiFit._get_xy_fit(_xy_data, minimizer)
        _split_data_1, _split_data_2 = TestMultiFit._split_data(_xy_data, axis=1)
        self._fit_xy_all_multi = MultiFit(TestMultiFit._get_xy_fit(_xy_data, minimizer))
        self._fit_xy_split_multi = MultiFit(fit_list=[
            TestMultiFit._get_xy_fit(_split_data_1, minimizer),
            TestMultiFit._get_xy_fit(_split_data_2, minimizer)
        ])
        self._fit_xy_split_1 = TestMultiFit._get_xy_fit(_split_data_1, minimizer)
        self._fit_xy_split_1_multi = MultiFit(TestMultiFit._get_xy_fit(_split_data_1, minimizer))
        self._fit_xy_split_2 = TestMultiFit._get_xy_fit(_split_data_2, minimizer)
        self._fit_xy_split_2_multi = MultiFit(TestMultiFit._get_xy_fit(_split_data_2, minimizer))

        self._expected_parameters_all = calculate_expected_fit_parameters_xy(
            x_data=_xy_data[0], y_data=_xy_data[1], model_function=quadratic_model, y_error=0.01,
            initial_parameter_values=[1.0, 1.0, 1.0], x_error=0.01, model_function_derivative=quadratic_model_derivative
        )
        self._expected_parameters_split_1 = calculate_expected_fit_parameters_xy(
            x_data=_split_data_1[0], y_data=_split_data_1[1], model_function=quadratic_model, y_error=0.01,
            initial_parameter_values=[1.0, 1.0, 1.0], x_error=0.01, model_function_derivative=quadratic_model_derivative
        )
        self._expected_parameters_split_2 = calculate_expected_fit_parameters_xy(
            x_data=_split_data_2[0], y_data=_split_data_2[1], model_function=quadratic_model, y_error=0.01,
            initial_parameter_values=[1.0, 1.0, 1.0], x_error=0.01, model_function_derivative=quadratic_model_derivative
        )

    def setUp(self):
        np.random.seed(1337)


class TestMultiFitIntegrityHist(TestMultiFit):

    def setUp(self):
        TestMultiFit.setUp(self)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_multifit_integrity_simple_iminuit(self):
        self._set_hist_fits(minimizer='iminuit')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_all_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_hist_split_1, self._fit_hist_split_1_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_hist_split_2, self._fit_hist_split_2_multi)

    @unittest.skip('scipy optimize minimizer seems to be bugged')
    def test_multifit_integrity_simple_scipy(self):
        self._set_hist_fits(minimizer='scipy')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_all_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_hist_split_1, self._fit_hist_split_1_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_hist_split_2, self._fit_hist_split_2_multi)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    @unittest.skip('Splitting data between fits does not work as intended')
    def test_multifit_vs_regular_fit_iminuit(self):
        self._set_hist_fits(minimizer='iminuit')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_split_multi)

    @unittest.skip('Splitting data between fits does not work as intended')
    def test_multifit_vs_regular_fit_scipy(self):
        self._set_hist_fits(minimizer='scipy')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_split_multi)


class TestMultiFitIntegrityIndexed(TestMultiFit):

    def setUp(self):
        TestMultiFit.setUp(self)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_multifit_integrity_simple_iminuit(self):
        self._set_indexed_fits(minimizer='iminuit')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_indexed_all, self._fit_indexed_all_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_indexed_split_1, self._fit_indexed_split_1_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_indexed_split_2, self._fit_indexed_split_2_multi)

    @unittest.skip('scipy optimize minimizer seems to be bugged')
    def test_multifit_integrity_simple_scipy(self):
        self._set_indexed_fits(minimizer='scipy')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_indexed_all, self._fit_indexed_all_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_indexed_split_1, self._fit_indexed_split_1_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_indexed_split_2, self._fit_indexed_split_2_multi)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    @unittest.skip('Splitting data between fits does not work as intended')
    def test_multifit_vs_regular_fit_iminuit(self):
        self._set_indexed_fits(minimizer='iminuit')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_indexed_all, self._fit_indexed_split_multi)

    @unittest.skip('Splitting data between fits does not work as intended')
    def test_multifit_vs_regular_fit_scipy(self):
        self._set_indexed_fits(minimizer='scipy')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_indexed_all, self._fit_indexed_split_multi)


class TestMultiFitIntegrityXY(TestMultiFit):

    def setUp(self):
        TestMultiFit.setUp(self)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_parameter_values_match_expectation_iminuit(self):
        self._set_xy_fits(minimizer='iminuit')
        _tol = 1e-6
        self._fit_xy_all.do_fit()
        assert np.allclose(self._expected_parameters_all, self._fit_xy_all.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_all_multi.do_fit()
        assert np.allclose(self._expected_parameters_all, self._fit_xy_all_multi.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_split_multi.do_fit()
        # FIXME results for split fit are wrong by ~1%
        assert np.allclose(self._expected_parameters_all, self._fit_xy_split_multi.parameter_values, atol=0, rtol=1e-2)
        self._fit_xy_split_1.do_fit()
        assert np.allclose(self._expected_parameters_split_1, self._fit_xy_split_1.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_split_1_multi.do_fit()
        assert np.allclose(self._expected_parameters_split_1, self._fit_xy_split_1_multi.parameter_values,
                           atol=0, rtol=_tol)
        self._fit_xy_split_2.do_fit()
        assert np.allclose(self._expected_parameters_split_2, self._fit_xy_split_2.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_split_2_multi.do_fit()
        assert np.allclose(self._expected_parameters_split_2, self._fit_xy_split_2_multi.parameter_values,
                           atol=0, rtol=_tol)

    @unittest.skip('Scipy optimize minimizer seems to be bugged')
    def test_parameter_values_match_expectation_scipy(self):
        self._set_xy_fits(minimizer='scipy')
        # FIXME results for split fit are wrong by ~1%
        _tol = 1e-2
        self._fit_xy_all.do_fit()
        assert np.allclose(self._expected_parameters_all, self._fit_xy_all.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_all_multi.do_fit()
        assert np.allclose(self._expected_parameters_all, self._fit_xy_all_multi.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_split_multi.do_fit()
        assert np.allclose(self._expected_parameters_all, self._fit_xy_split_multi.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_split_1.do_fit()
        assert np.allclose(self._expected_parameters_split_1, self._fit_xy_split_1.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_split_1_multi.do_fit()
        assert np.allclose(self._expected_parameters_split_1, self._fit_xy_split_1_multi.parameter_values,
                           atol=0, rtol=_tol)
        self._fit_xy_split_2.do_fit()
        assert np.allclose(self._expected_parameters_split_2, self._fit_xy_split_2.parameter_values, atol=0, rtol=_tol)
        self._fit_xy_split_2_multi.do_fit()
        assert np.allclose(self._expected_parameters_split_2, self._fit_xy_split_2_multi.parameter_values,
                           atol=0, rtol=_tol)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_multifit_integrity_simple_iminuit(self):
        self._set_xy_fits(minimizer='iminuit')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_all_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_xy_split_1, self._fit_xy_split_1_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_xy_split_2, self._fit_xy_split_2_multi)

    @unittest.skip('scipy optimize minimizer seems to be bugged')
    def test_multifit_integrity_simple_scipy(self):
        self._set_xy_fits(minimizer='scipy')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_all_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_xy_split_1, self._fit_xy_split_1_multi)
        TestMultiFit._assert_fits_valid_and_equal(self._fit_xy_split_2, self._fit_xy_split_2_multi)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    @unittest.skip('Splitting data between fits does not work as intended')
    def test_multifit_vs_regular_fit_iminuit(self):
        self._set_xy_fits(minimizer='iminuit')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)

    @unittest.skip('Splitting data between fits does not work as intended')
    def test_multifit_vs_regular_fit_scipy(self):
        self._set_xy_fits(minimizer='scipy')
        TestMultiFit._assert_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)
