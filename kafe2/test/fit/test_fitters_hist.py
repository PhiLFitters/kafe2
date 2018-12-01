import unittest
import numpy as np
import six

from scipy import stats

from kafe2.config import kc
from kafe2.fit import HistContainer, HistFit
from kafe2.fit.histogram.fit import HistFitException
from kafe2.fit.histogram.model import HistModelFunctionException


CONFIG_PARAMETER_DEFAULT_VALUE = kc('core', 'default_initial_parameter_value')


class TestFittersHist(unittest.TestCase):

    @staticmethod
    def hist_model_density(x, mu=14., sigma=3.):
        return stats.norm(mu, sigma).pdf(x)

    @staticmethod
    def hist_model_density_antideriv(x, mu=14., sigma=3.):
        return stats.norm(mu, sigma).cdf(x)

    @staticmethod
    def simple_chi2(data, model):
        return np.sum((data - model)**2)


    def setUp(self):
        self._ref_pm_support = np.linspace(-5, 5, 11)

        self._ref_n_bins = 11
        self._ref_n_bin_range = (-3, 25)
        self._ref_bin_edges = np.linspace(self._ref_n_bin_range[0], self._ref_n_bin_range[1], self._ref_n_bins+1)

        self._ref_n_entries = 100
        self._ref_params = (14., 3.)

        self._ref_entries = np.array([ 11.47195963,   9.96715403,  19.90275216,  13.65225802,
                                    18.52670233,  17.79707059,  11.5441954 ,  18.42331074,
                                    13.3808496 ,  18.40632518,  13.21694177,  15.34569261,
                                     9.85164713,  11.56275047,  12.42687109,   9.43924719,
                                    15.03632673,  10.76203991,  15.16788754,  13.78683534,
                                    10.84686619,  17.29655143,  12.58404203,  13.10171453,
                                    20.23773075,  15.00936796,  15.38972301,  12.94254624,
                                    19.01920031,  19.67747312,  14.53196033,  13.78898329,
                                    17.34040707,  17.25328064,  13.64052295,  11.80461841,
                                    14.21872985,  11.59054406,  20.28243216,  15.43544627,
                                    15.85964469,  13.68948162,  14.55574162,  15.70198922,
                                    12.85548112,  14.05419407,  15.85172907,  19.2951937 ,
                                    12.70652203,  13.47822913,  12.84301792,  13.65407699,
                                    12.81908235,  12.73065184,  23.34917367,  11.38895625,
                                     9.41615435,   7.24554349,  11.4112971 ,  14.54736404,
                                    15.84186607,  15.33880379,   9.93177073,  15.14220967,
                                    15.19535013,  11.00033758,  11.83671083,  13.02411982,
                                    18.43323913,  12.06272124,  20.57549426,  12.97405382,
                                    13.35969913,  10.76072424,  16.35912216,  14.52842827,
                                    16.97221766,  13.02791932,  14.36097113,  16.93839498,
                                    15.60091018,  11.55044489,  14.06915886,  17.64985576,
                                    11.59865691,  13.41486068,  14.25999508,  10.70887047,
                                     4.08280244,  13.1043861 ,  14.62321312,  14.85894591,
                                    14.89235398,  10.60967181,  15.22310211,  10.77853626,
                                    14.56823312,  14.46093346,  13.34031129,  14.14203599])

        self._ref_model = (self.hist_model_density_antideriv(self._ref_bin_edges[1:], *self._ref_params) -
                          self.hist_model_density_antideriv(self._ref_bin_edges[:-1], *self._ref_params)) * self._ref_n_entries

        #print map(int, self._ref_entries)

        self._ref_hist_cont = HistContainer(self._ref_n_bins, self._ref_n_bin_range, bin_edges=None, fill_data=self._ref_entries)

        self.hist_fit = HistFit(data=self._ref_hist_cont,
                                model_density_function=self.hist_model_density,
                                cost_function=self.simple_chi2,
                                model_density_antiderivative=self.hist_model_density_antideriv)
        self.hist_fit.add_simple_error(err_val=1.0)
        self.hist_fit_default_cost_function = HistFit(data=self._ref_hist_cont,
                                                      model_density_function=self.hist_model_density,
                                                      model_density_antiderivative=self.hist_model_density_antideriv)
        self.hist_fit_default_cost_function.add_simple_error(err_val=1.0)

        self._ref_parameter_value_estimates = [13.828005427495496, 2.6276452391799703]
        self._ref_parameter_value_estimates_default_cost_function = [14.185468816590726, 3.0232973450410165]
        self._ref_model_estimates = (self.hist_model_density_antideriv(self._ref_bin_edges[1:], *self._ref_parameter_value_estimates) -
                                     self.hist_model_density_antideriv(self._ref_bin_edges[:-1], *self._ref_parameter_value_estimates)) * self._ref_n_entries
        self._ref_model_estimates_default_cost_function = (self.hist_model_density_antideriv(self._ref_bin_edges[1:], *self._ref_parameter_value_estimates_default_cost_function) -
                                                           self.hist_model_density_antideriv(self._ref_bin_edges[:-1], *self._ref_parameter_value_estimates_default_cost_function)) * self._ref_n_entries


    def test_before_fit_compare_parameter_values(self):
        self.assertTrue(
            np.allclose(
                self.hist_fit.parameter_values,
                self._ref_params
            )
        )

    def test_before_fit_compare_model_values(self):
        self.assertTrue(
            np.allclose(
                self.hist_fit.model,
                self._ref_model,
                rtol=1e-2
            )
        )

    def test_do_fit_compare_parameter_values(self):
        self.hist_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.hist_fit.parameter_values,
                self._ref_parameter_value_estimates,
                rtol=1e-3
            )
        )

    def test_do_fit_compare_model_values(self):
        self.hist_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.hist_fit.model,
                self._ref_model_estimates,
                rtol=1e-2
            )
        )

    def test_do_fit_default_cost_function_compare_parameter_values(self):
        self.hist_fit_default_cost_function.do_fit()
        self.assertTrue(
            np.allclose(
                self.hist_fit_default_cost_function.parameter_values,
                self._ref_parameter_value_estimates_default_cost_function,
                rtol=1e-3
            )
        )

    def test_do_fit_default_cost_function_compare_model_values(self):
        self.hist_fit_default_cost_function.do_fit()
        self.assertTrue(
            np.allclose(
                self.hist_fit_default_cost_function.model,
                self._ref_model_estimates_default_cost_function,
                rtol=1e-2
            )
        )

    def test_update_cost_function_on_parameter_change(self):
        self.hist_fit.set_all_parameter_values(self._ref_parameter_value_estimates)
        self.assertEqual(
            self.hist_fit.cost_function_value,
            self.hist_fit._cost_function(self.hist_fit.data, self._ref_model_estimates),
        )

    def test_report_before_fit(self):
        _buffer = six.StringIO()
        self.hist_fit.report(output_stream=_buffer)
        self.assertNotEquals(_buffer.getvalue(), "")

    def test_report_after_fit(self):
        _buffer = six.StringIO()
        self.hist_fit.do_fit()
        self.hist_fit.report(output_stream=_buffer)
        self.assertNotEquals(_buffer.getvalue(), "")