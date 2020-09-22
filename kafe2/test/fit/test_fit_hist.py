import abc
import unittest2 as unittest
import numpy as np
import six

from scipy import stats

from kafe2.core.minimizers import AVAILABLE_MINIMIZERS
from kafe2.core.fitters import NexusFitterException

from kafe2.config import kc

from kafe2.fit import HistFit, HistContainer
from kafe2.fit.histogram.fit import HistFitException
from kafe2.fit.histogram.model import HistModelFunctionException, HistParametricModelException
from kafe2.fit.histogram.cost import HistCostFunction_NegLogLikelihood

from kafe2.test.fit.test_fit import AbstractTestFit


def simple_chi2(data, model):
    return np.sum((data - model)**2)


def hist_model_density(x, mu=14., sigma=3.):
    return stats.norm(mu, sigma).pdf(x)

def hist_model_density_antideriv(x, mu=14., sigma=3.):
    return stats.norm(mu, sigma).cdf(x)



class TestHistFitBasicInterface(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        self._ref_n_bins = 11
        self._ref_n_bin_range = (-3, 25)
        self._ref_bin_edges = np.linspace(self._ref_n_bin_range[0], self._ref_n_bin_range[1], self._ref_n_bins+1)

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

        self._ref_hist_cont = HistContainer(
            self._ref_n_bins, self._ref_n_bin_range, bin_edges=None, fill_data=self._ref_entries)


        # reference initial values
        self._ref_initial_pars = np.array([14., 3.])
        self._ref_initial_model = (
            hist_model_density_antideriv(self._ref_bin_edges[1:], *self._ref_initial_pars) -
            hist_model_density_antideriv(self._ref_bin_edges[:-1], *self._ref_initial_pars)) * len(self._ref_entries)

        # fit data
        self._ref_data, _ = np.histogram(
            self._ref_entries,
            bins=self._ref_n_bins,
            range=self._ref_n_bin_range)

        # pre-fit cost value
        self._ref_initial_cost_nll = \
            -2*np.sum(stats.poisson.logpmf(self._ref_data, self._ref_initial_model))
        self._ref_initial_cost_chi2 = simple_chi2(self._ref_data, self._ref_initial_model)

        # reference fit result values
        self._nominal_fit_result_pars_nll = np.array([14.18427759, 3.02257722])
        self._nominal_fit_result_pars_chi2 = np.array([13.82779489, 2.62746457])

        self._nominal_fit_result_model_nll = (
            hist_model_density_antideriv(self._ref_bin_edges[1:], *self._nominal_fit_result_pars_nll) -
            hist_model_density_antideriv(self._ref_bin_edges[:-1], *self._nominal_fit_result_pars_nll)) * len(self._ref_entries)
        self._nominal_fit_result_model_chi2 = (
            hist_model_density_antideriv(self._ref_bin_edges[1:], *self._nominal_fit_result_pars_chi2) -
            hist_model_density_antideriv(self._ref_bin_edges[:-1], *self._nominal_fit_result_pars_chi2)) * len(self._ref_entries)

        self._nominal_fit_result_cost_nll = \
            -2*np.sum(stats.poisson.logpmf(self._ref_data, self._nominal_fit_result_model_nll))
        self._nominal_fit_result_cost_chi2 = simple_chi2(self._ref_data, self._nominal_fit_result_model_chi2)

        # helper dict with all reference property values
        self._ref_prop_dict = dict(
            did_fit=False,
            model_count=1,

            parameter_values=self._ref_initial_pars,
            parameter_names=('mu', 'sigma'),

            data=self._ref_data,
            model=self._ref_initial_model,
        )

    def _get_fit(self, model_density_function=None, bin_evaluation="numerical", cost_function=None):
        '''convenience'''

        model_density_function = model_density_function or hist_model_density

        # TODO: fix default
        cost_function = cost_function or HistCostFunction_NegLogLikelihood(
            data_point_distribution='poisson')

        _fit = HistFit(
            data=self._ref_hist_cont,
            model_density_function=model_density_function,
            bin_evaluation=bin_evaluation,
            cost_function=cost_function,
            minimizer=self.MINIMIZER
        )
        _fit.add_error(1.0)  # only considered for chi2

        return _fit

    def _get_test_fits(self):
        return {
            # numeric integration takes too long for testing
            #'default': \
            #    self._get_fit(),
            'explicit_chi2': \
                self._get_fit(cost_function=simple_chi2, bin_evaluation=hist_model_density_antideriv),
            'model_with_antiderivative': \
                self._get_fit(bin_evaluation=hist_model_density_antideriv),
        }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                cost_function_value=np.float64(self._ref_initial_cost_nll),  # TODO: fix type
            ),
            fit_names=['default', 'model_with_antiderivative']
        )
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                cost_function_value=np.float64(self._ref_initial_cost_chi2),  # TODO: fix type
            ),
            fit_names=['explicit_chi2']
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars_nll,
                model=self._nominal_fit_result_model_nll,
                did_fit=True,
                cost_function_value=np.float64(self._nominal_fit_result_cost_nll),
            ),
            fit_names=['default', 'model_with_antiderivative'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-2
        )
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars_chi2,
                model=self._nominal_fit_result_model_chi2,
                did_fit=True,
                cost_function_value=np.float64(self._nominal_fit_result_cost_chi2),
            ),
            fit_names=['explicit_chi2'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-2
        )

    def test_update_data(self):
        _fit = self._get_fit(bin_evaluation=hist_model_density_antideriv)

        _new_entries = np.array([
            19.424357258680693, 19.759361803397155, 18.364336396273362, 18.36464562195573,
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

        _fit.data = HistContainer(
            8,
            (17, 23),
            fill_data=_new_entries
        )
        _fit.add_error(err_val=1.0)

        _ref_data, _ = np.histogram(
            _new_entries,
            bins=8,
            range=(17, 23)
        )

        self._assert_fit_properties(
            _fit,
            dict(
                data=_ref_data,
            )
        )

        _fit.do_fit()

        self._assert_fit_properties(
            _fit,
            dict(
                data=_ref_data,
                parameter_values=np.array([19.83815938,  1.1729322]),
            ),
            rtol=1e-2
        )

    def test_reserved_parameter_names_raise(self):
        def dummy_model(x, data):
            pass

        with self.assertRaises(HistFitException) as _exc:
            HistFit(data=self._ref_hist_cont,
                    model_density_function=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn('reserved', _exc.exception.args[0])
        self.assertIn('data', _exc.exception.args[0])

    def test_model_no_pars_raise(self):
        def dummy_model():
            pass

        with self.assertRaises(HistModelFunctionException) as _exc:
            HistFit(data=self._ref_hist_cont,
                    model_density_function=dummy_model,
                    bin_evaluation=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn(
            "needs at least one parameter",
            _exc.exception.args[0])

    def test_model_no_pars_beside_x_raise(self):
        def dummy_model(x):
            pass

        with self.assertRaises(HistModelFunctionException) as _exc:
            HistFit(data=self._ref_hist_cont,
                    model_density_function=dummy_model,
                    bin_evaluation=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn(
            "needs at least one parameter besides",
            _exc.exception.args[0])

    def test_model_varargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, *varargs):
            pass

        with self.assertRaises(HistModelFunctionException) as _exc:
            HistFit(data=self._ref_hist_cont,
                    model_density_function=dummy_model,
                    bin_evaluation=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varargs', _exc.exception.args[0])

    def test_model_varkwargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, **varkwargs):
            pass

        with self.assertRaises(HistModelFunctionException) as _exc:
            HistFit(data=self._ref_hist_cont,
                    model_density_function=dummy_model,
                    bin_evaluation=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varkwargs', _exc.exception.args[0])

    def test_model_varargs_varkwargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, *varargs, **varkwargs):
            pass

        with self.assertRaises(HistModelFunctionException) as _exc:
            HistFit(data=self._ref_hist_cont,
                    model_density_function=dummy_model,
                    bin_evaluation=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varargs', _exc.exception.args[0])
        # TODO: enable when implemented
        #self.assertIn('varkwargs', _exc.exception.args[0])

    def test_model_and_antiderivative_different_signatures_raise(self):
        def dummy_model(x, mu, sigma):
            pass

        def dummy_model_antiderivative(x, mu, bogus):
            pass

        with self.assertRaises(ValueError) as _exc:
            HistFit(data=self._ref_hist_cont,
                    model_density_function=dummy_model,
                    bin_evaluation=dummy_model_antiderivative,
                    minimizer=self.MINIMIZER)

    def test_model_and_antiderivative_no_defaults(self):
        def legendre_grade_2(x, a=1, b=2, c=3):
            return a + b * x + c * 0.5 * (3 * x ** 2 - 1)

        def legendre_grade_2_integrated(x, a, b, c):
            return 0.5 * x * (2 * a + b * x + c * (x ** 2 - 1))

        # should not raise an error
        HistFit(data=self._ref_hist_cont,
                model_density_function=legendre_grade_2,
                bin_evaluation=legendre_grade_2_integrated,
                minimizer=self.MINIMIZER)

    def test_report_before_fit(self):
        # TODO: check report content
        _buffer = six.StringIO()
        _fit = self._get_fit(bin_evaluation=hist_model_density_antideriv)
        _fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")

    def test_report_after_fit(self):
        # TODO: check report content
        _buffer = six.StringIO()
        _fit = self._get_fit(bin_evaluation=hist_model_density_antideriv)
        _fit.do_fit()
        _fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")
