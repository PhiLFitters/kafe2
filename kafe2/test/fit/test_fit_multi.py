import numpy as np
from scipy.stats import norm
import unittest2 as unittest

from kafe2 import HistContainer, HistFit, MultiFit, XYFit

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
    def _get_xy_data(err_x=0.01, err_y=0.01, a=1.2, b=2.3, c=3.4):
        _x_0 = np.linspace(start=-10, stop=10, num=101, endpoint=True)
        _x_jitter = np.random.normal(loc=0, scale=err_x, size=101)
        _x_data = _x_0 + _x_jitter

        _y_0 = TestMultiFit.quadratic_model(_x_data, a, b, c)
        _y_jitter = np.random.normal(loc=0, scale=err_y, size=101)
        _y_data = _y_0 + _y_jitter

        return np.array([_x_data, _y_data])

    @staticmethod
    def quadratic_model(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def _get_xy_fit(xy_data, minimizer, err_x=0.01, err_y=0.01):
        _xy_fit = XYFit(
            xy_data=xy_data,
            model_function=TestMultiFit.quadratic_model,
            cost_function='chi2',
            minimizer=minimizer,
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


    def setUp(self):
        np.random.seed(1337)


class TestMultiFitIntegrityHist(TestMultiFit):

    @staticmethod
    def _assert_hist_fits_valid_and_equal(_hist_fit_1, _hist_fit_2):
        _hist_fit_1.do_fit()
        _hist_fit_2.do_fit()

        assert _hist_fit_1.parameter_values[0] != 1.0
        assert _hist_fit_1.parameter_values[1] != 1.0
        assert _hist_fit_2.parameter_values[0] != 1.0
        assert _hist_fit_2.parameter_values[1] != 1.0
        _tol = 1e-6
        assert np.allclose(_hist_fit_1.parameter_values, _hist_fit_2.parameter_values, atol=0, rtol=_tol)
        assert np.allclose(_hist_fit_1.parameter_errors, _hist_fit_2.parameter_errors, atol=0, rtol=_tol)
        assert _hist_fit_1.parameter_cor_mat is not None
        assert _hist_fit_2.parameter_cor_mat is not None
        assert np.allclose(_hist_fit_1.parameter_cor_mat, _hist_fit_2.parameter_cor_mat, atol=0, rtol=_tol)
        assert _hist_fit_1.parameter_cov_mat is not None
        assert _hist_fit_2.parameter_cov_mat is not None
        assert np.allclose(_hist_fit_1.parameter_cov_mat, _hist_fit_2.parameter_cov_mat, atol=0, rtol=_tol)

    def setUp(self):
        TestMultiFit.setUp(self)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_multifit_integrity_simple_iminuit(self):
        self._set_hist_fits(minimizer='iminuit')
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_all_multi)
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_hist_split_1, self._fit_hist_split_1_multi)
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_hist_split_2, self._fit_hist_split_2_multi)

    def test_multifit_integrity_simple_scipy(self):
        self._set_hist_fits(minimizer='scipy')
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_all_multi)
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_hist_split_1, self._fit_hist_split_1_multi)
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_hist_split_2, self._fit_hist_split_2_multi)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_multifit_vs_regular_fit_iminuit(self):
        self._set_hist_fits(minimizer='iminuit')
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_split_multi)

    def test_multifit_vs_regular_fit_scipy(self):
        self._set_hist_fits(minimizer='scipy')
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_hist_all, self._fit_hist_split_multi)


class TestMultiFitIntegrityXY(TestMultiFit):

    @staticmethod
    def _assert_xy_fits_valid_and_equal(_xy_fit_1, _xy_fit_2):
        _xy_fit_1.do_fit()
        _xy_fit_2.do_fit()

        assert _xy_fit_1.parameter_values[0] != 1.0
        assert _xy_fit_1.parameter_values[1] != 1.0
        assert _xy_fit_1.parameter_values[2] != 1.0
        assert _xy_fit_2.parameter_values[0] != 1.0
        assert _xy_fit_2.parameter_values[1] != 1.0
        assert _xy_fit_2.parameter_values[2] != 1.0
        _tol = 1e-6
        assert not np.any(np.isnan(_xy_fit_1.parameter_values))
        assert not np.any(np.isnan(_xy_fit_2.parameter_values))
        assert np.allclose(_xy_fit_1.parameter_values, _xy_fit_2.parameter_values, atol=0, rtol=_tol)
        assert not np.any(np.isnan(_xy_fit_1.parameter_errors))
        assert not np.any(np.isnan(_xy_fit_2.parameter_errors))
        assert np.allclose(_xy_fit_1.parameter_errors, _xy_fit_2.parameter_errors, atol=0, rtol=_tol)
        assert _xy_fit_1.parameter_cor_mat is not None
        assert _xy_fit_2.parameter_cor_mat is not None
        assert np.allclose(_xy_fit_1.parameter_cor_mat, _xy_fit_2.parameter_cor_mat, atol=0, rtol=_tol)
        assert _xy_fit_1.parameter_cov_mat is not None
        assert _xy_fit_2.parameter_cov_mat is not None
        assert np.allclose(_xy_fit_1.parameter_cov_mat, _xy_fit_2.parameter_cov_mat, atol=0, rtol=_tol)

    def setUp(self):
        TestMultiFit.setUp(self)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_multifit_integrity_simple_iminuit(self):
        self._set_xy_fits(minimizer='iminuit')
        TestMultiFitIntegrityXY._assert_xy_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_all_multi)
        TestMultiFitIntegrityXY._assert_xy_fits_valid_and_equal(self._fit_xy_split_1, self._fit_xy_split_1_multi)
        TestMultiFitIntegrityXY._assert_xy_fits_valid_and_equal(self._fit_xy_split_2, self._fit_xy_split_2_multi)

    def test_multifit_integrity_simple_scipy(self):
        self._set_xy_fits(minimizer='scipy')
        TestMultiFitIntegrityXY._assert_xy_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_all_multi)
        TestMultiFitIntegrityXY._assert_xy_fits_valid_and_equal(self._fit_xy_split_1, self._fit_xy_split_1_multi)
        TestMultiFitIntegrityXY._assert_xy_fits_valid_and_equal(self._fit_xy_split_2, self._fit_xy_split_2_multi)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_multifit_vs_regular_fit_iminuit(self):
        self._set_xy_fits(minimizer='iminuit')
        TestMultiFitIntegrityXY._assert_xy_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)

    def test_multifit_vs_regular_fit_scipy(self):
        self._set_xy_fits(minimizer='scipy')
        TestMultiFitIntegrityXY._assert_xy_fits_valid_and_equal(self._fit_xy_all, self._fit_xy_split_multi)
