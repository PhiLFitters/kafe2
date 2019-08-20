import unittest
import numpy as np
import six

from scipy import stats

from kafe2.config import kc
from kafe2.fit import HistContainer, HistFit
from kafe2.fit.histogram.fit import HistFitException
from kafe2.fit.histogram.model import HistModelFunctionException


CONFIG_PARAMETER_DEFAULT_VALUE = kc('core', 'default_initial_parameter_value')
DEFAULT_TEST_MINIMIZER = 'scipy'


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
        self._ref_n_bins_2 = 8
        self._ref_n_bin_range = (-3, 25)
        self._ref_n_bin_range_2 = (17, 23)
        self._ref_bin_edges = np.linspace(self._ref_n_bin_range[0], self._ref_n_bin_range[1], self._ref_n_bins+1)

        self._ref_n_entries = 100
        self._ref_params = (14., 3.)
        self._ref_params_2 = (20., 1.)

        self._ref_entries = np.array([11.47195963,   9.96715403,  19.90275216,  13.65225802,
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

        self._ref_entries_2 = np.array([19.928222745776985, 19.639279266737375, 20.514809933701105, 21.127041800930975,
                                        19.431340012679225, 18.221578358444123, 19.84374096736842, 20.717108873125888,
                                        18.29969637477776, 19.636904203431605, 20.598572474665936, 19.051704032334314,
                                        20.397295019394786, 17.771329618244778, 19.793657664134, 20.606425366995445,
                                        18.03539465458286, 21.24370157890459, 19.126411526517344, 19.489650369914784,
                                        19.143992764240835, 18.273618489220993, 20.579479369020582, 20.764146679134214,
                                        21.710409155277947, 19.90356794567859, 20.529936878297015, 19.425295611964422,
                                        20.977463581294952, 20.079993122787975, 20.903685193957756, 21.395419976959257,
                                        19.632702640947038, 21.168167853114134, 21.028238701226933, 22.289168075231366,
                                        20.883454030240475, 19.867637550898262, 21.172121792691495, 20.06738826819919,
                                        20.308579479424964, 19.19121302573849, 18.755470698579806, 21.114095139148475,
                                        19.59493893613369, 19.651815793716445, 19.62154711580933, 18.55014170015994,
                                        22.526449608297227, 18.110390138012633, 21.002743128628367, 21.29527749783569,
                                        20.221850737188696, 20.76505911646573, 19.477645665151954, 19.60394959853505,
                                        18.656819850990313, 17.663335978786627, 20.735143798012814, 18.76234983324196,
                                        19.337590804907336, 20.701249661772515, 20.67823676826869, 18.151585289698666,
                                        21.03879990944658, 21.633074130209437, 18.40559099568937, 18.81230418040524,
                                        20.993431209576997, 19.992803944850124, 22.0896955555629, 18.57269651622513,
                                        20.780848264778246, 19.322037240156526, 20.556160946991504, 19.047366020241302,
                                        21.420061893722288, 17.901786331439553, 18.301912704050217, 18.47842163635392,
                                        20.419867968944835, 22.078940310248505, 19.263495204696028, 19.160841114563702,
                                        18.810868144238153, 21.321006818824664, 20.679797214847394, 19.612398935778216,
                                        20.80243155270663, 19.643735868737544, 20.717679814765503, 19.740543205809963,
                                        20.11580480227135, 21.261233879958397, 19.176543378780835, 19.968799388911513,
                                        19.452523885698803, 21.042616672191393, 19.94794141542776, 19.87054647329419])

        self._ref_model = (self.hist_model_density_antideriv(self._ref_bin_edges[1:], *self._ref_params) -
                          self.hist_model_density_antideriv(self._ref_bin_edges[:-1], *self._ref_params)) * self._ref_n_entries

        #print map(int, self._ref_entries)

        self._ref_hist_cont = HistContainer(self._ref_n_bins, self._ref_n_bin_range, bin_edges=None, fill_data=self._ref_entries)

        self.hist_fit = HistFit(data=self._ref_hist_cont,
                                model_density_function=self.hist_model_density,
                                cost_function=self.simple_chi2,
                                model_density_antiderivative=self.hist_model_density_antideriv,
                                minimizer=DEFAULT_TEST_MINIMIZER)
        self.hist_fit.add_simple_error(err_val=1.0)
        self.hist_fit_default_cost_function = HistFit(data=self._ref_hist_cont,
                                                      model_density_function=self.hist_model_density,
                                                      model_density_antiderivative=self.hist_model_density_antideriv,
                                                      minimizer=DEFAULT_TEST_MINIMIZER)
        self.hist_fit_default_cost_function.add_simple_error(err_val=1.0)

        self._ref_parameter_value_estimates = [13.82779355, 2.62031141]
        self._ref_parameter_value_estimates_default_cost_function = [14.18443871, 3.0148702]
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

    def test_nexus_update_on_data_change(self):
        # TODO: when setting new data with a different length has been fixed, change the size of the new data to test
        #       this as well
        new_data = HistContainer(self._ref_n_bins_2, self._ref_n_bin_range_2, bin_edges=None,
                                 fill_data=self._ref_entries_2)
        self.hist_fit.data = new_data
        self.hist_fit.add_simple_error(err_val=1.0)
        self.hist_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.hist_fit.parameter_values,
                self._ref_params_2,
                atol=0.2
            )
        )
