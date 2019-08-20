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

        self._ref_entries_2 = np.array([19.424357258680693, 19.759361803397155, 18.364336396273362, 18.36464562195573,
                                        21.12001399388072, 18.08232152646234, 20.881797998050466, 19.799071480564336,
                                        20.00149234521563, 18.45879580610016, 20.507568873033875, 20.30648149729429,
                                        20.877350386469818, 19.381958240681676, 19.64442840120083, 19.141355147035597,
                                        19.94101630498644, 20.504982848860507, 17.524033521076785, 20.73543156774036,
                                        21.163113940992737, 19.05979241568689, 20.196952015154917, 20.40130402635463,
                                        21.80417387186999, 20.611530513961164, 19.099481246155072, 21.817653009909904,
                                        19.75055842274471, 20.103815812928232, 18.174017677733538, 21.047249981725685,
                                        21.262823911834488, 21.536864685525888, 18.813610758324447, 21.499655806695163,
                                        19.933264932657465, 21.954933746841995, 17.78470283680212, 19.8917212343489,
                                        19.372012624773838, 18.9520723656944, 19.9905553737993, 18.22737716365809,
                                        22.208437406472243, 19.875706306083835, 19.17672889225326, 20.10750939196147,
                                        20.093938177045032, 19.857292210131092, 20.17843836897936, 20.58803422718744,
                                        19.936410829343984, 19.050688989087522, 18.46936492682146, 21.90955106395087,
                                        19.661176212242154, 22.2764766192496, 19.850200163818528, 18.49289303805954,
                                        19.7563960302135, 20.940311019530235, 19.12732791777932, 22.09224225083453,
                                        20.225667564052465, 20.10787811564912, 18.660130651239726, 18.356069221596094,
                                        20.278651217320608, 18.62176541545302, 18.747451690981315, 19.81307693501857,
                                        19.34065619310232, 19.56998674371285, 19.885577923257177, 18.81752399043877,
                                        20.67686318083984, 20.265021790145465, 19.982547007042093, 19.581967230877964,
                                        18.486722000426457, 19.83143305661045, 21.252124382516378, 20.152988937293436,
                                        18.917354464336892, 18.349803731030892, 21.32702043081207, 22.410955706069743,
                                        20.972404800516973, 19.615870251101295, 19.013627387925588, 19.54487437668081,
                                        20.538542465210206, 18.626198427902466, 20.221745437611307, 19.064952809088076])

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
        new_data = HistContainer(self._ref_n_bins_2, self._ref_n_bin_range_2, bin_edges=None,
                                 fill_data=self._ref_entries_2)
        self.hist_fit.data = new_data
        self.hist_fit.add_simple_error(err_val=1.0)
        self.hist_fit.do_fit()
        print(self.hist_fit.parameter_values)
        self.assertTrue(
            np.allclose(
                self.hist_fit.parameter_values,
                self._ref_params_2,
                rtol=0.16
            )
        )
