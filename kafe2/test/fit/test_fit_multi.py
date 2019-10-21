import numpy as np
from scipy.stats import norm
import unittest2 as unittest

from kafe2 import HistContainer, HistFit, MultiFit

_cannot_import_IMinuit = False
try:
    from kafe2.core.minimizers.iminuit_minimizer import MinimizerIMinuit
except ImportError:
    _cannot_import_IMinuit = True


class TestMultiFit(unittest.TestCase):

    @staticmethod
    def _split_data(data):
        _random = np.random.rand(data.shape[0])
        _split_indices_1 = np.argwhere(_random < 0.5).flatten()
        _split_indices_2 = np.argwhere(_random >= 0.5).flatten()
        return data[_split_indices_1], data[_split_indices_2]

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
        self._fit_all = TestMultiFit._get_hist_fit(_raw_data, minimizer)
        _split_data_1, _split_data_2 = TestMultiFit._split_data(_raw_data)
        self._fit_all_multi = MultiFit(TestMultiFit._get_hist_fit(_raw_data, minimizer))
        self._fit_split_multi = MultiFit(fit_list=[
            TestMultiFit._get_hist_fit(_split_data_1, minimizer),
            TestMultiFit._get_hist_fit(_split_data_2, minimizer),
        ])
        self._fit_split_1 = TestMultiFit._get_hist_fit(_split_data_1, minimizer)
        self._fit_split_1_multi = MultiFit(TestMultiFit._get_hist_fit(_split_data_1, minimizer))
        self._fit_split_2 = TestMultiFit._get_hist_fit(_split_data_2, minimizer)
        self._fit_split_2_multi = MultiFit(TestMultiFit._get_hist_fit(_split_data_2, minimizer))

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
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_all, self._fit_all_multi)
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_split_1, self._fit_split_1_multi)
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_split_2, self._fit_split_2_multi)

    def test_multifit_integrity_simple_scipy(self):
        self._set_hist_fits(minimizer='scipy')
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_all, self._fit_all_multi)
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_split_1, self._fit_split_1_multi)
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_split_2, self._fit_split_2_multi)

    @unittest.skipIf(_cannot_import_IMinuit, 'Cannot import iminuit')
    def test_multifit_vs_regular_fit_iminuit(self):
        self._set_hist_fits(minimizer='iminuit')
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_all, self._fit_split_multi)

    def test_multifit_vs_regular_fit_scipy(self):
        self._set_hist_fits(minimizer='scipy')
        TestMultiFitIntegrityHist._assert_hist_fits_valid_and_equal(self._fit_all, self._fit_split_multi)