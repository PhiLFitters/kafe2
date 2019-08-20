import unittest
import numpy as np
import six

from scipy import stats

from kafe2.config import kc
from kafe2.fit import UnbinnedContainer, UnbinnedFit
from kafe2.fit.unbinned.fit import UnbinnedFitException


CONFIG_PARAMETER_DEFAULT_VALUE = kc('core', 'default_initial_parameter_value')
DEFAULT_TEST_MINIMIZER = 'scipy'


class TestFittersUnbinned(unittest.TestCase):

    @staticmethod
    def data_model_density(x, tau=2.2, fbg=0.1):
        b = 11.5
        a = 1.
        pdf1 = np.exp(-x / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
        pdf2 = 1. / (b - a)
        return (1 - fbg) * pdf1 + fbg * pdf2

    def setUp(self):
        self._ref_params = (2.2, 0.1)
        self._ref_params_2 = (2.0, 0.1)

        self._ref_data = np.array([7.42, 3.773, 5.968, 4.924, 1.468, 4.664, 1.745, 2.144, 3.836, 3.132, 1.568, 2.352,
                                   2.132, 9.381, 1.484, 1.181, 5.004, 3.06, 4.582, 2.076, 1.88, 1.337, 3.092, 2.265,
                                   1.208, 2.753, 4.457, 3.499, 8.192, 5.101, 1.572, 5.152, 4.181, 3.52, 1.344, 10.29,
                                   1.152, 2.348, 2.228, 2.172, 7.448, 1.108, 4.344, 2.042, 5.088, 1.02, 1.051, 1.987,
                                   1.935, 3.773, 4.092, 1.628, 1.688, 4.502, 4.687, 6.755, 2.56, 1.208, 2.649, 1.012,
                                   1.73, 2.164, 1.728, 4.646, 2.916, 1.101, 2.54, 1.02, 1.176, 4.716, 9.671, 1.692,
                                   9.292, 10.72, 2.164, 2.084, 2.616, 1.584, 5.236, 3.663, 3.624, 1.051, 1.544, 1.496,
                                   1.883, 1.92, 5.968, 5.89, 2.896, 2.76, 1.475, 2.644, 3.6, 5.324, 8.361, 3.052, 7.703,
                                   3.83, 1.444, 1.343])

        self._ref_data_2 = np.array([1.891075005713712, 2.361504517508682, 1.2026086768765802, 2.0192327689754013,
                                     6.789827691034451, 6.229060175471942, 5.6827777380476725, 1.3896314892970838,
                                     1.0379155165227294, 1.1948753489234705, 1.8735602750435354, 2.139305908923123,
                                     1.935777036301727, 6.108651728684855, 1.5185867568257783, 1.1244705951426477,
                                     2.2175887045709146, 3.900030680803669, 4.110502050796017, 3.634194797012612,
                                     4.931709433880977, 3.779552162832593, 3.6401901478360594, 1.7084479392384508,
                                     10.502957316458524, 1.669761159672512, 2.0378350020970446, 1.4406236513060318,
                                     2.3226997026175917, 1.2530908142355945, 2.4681349735479428, 2.2697496807176245,
                                     8.056140162306527, 6.483413760658378, 2.0103631281648733, 1.8484180369382175,
                                     2.042048282225629, 1.826991555740888, 4.119685425461164, 2.8467560883558485,
                                     1.8277818983512852, 9.894627016335363, 1.0841480000087294, 3.1732616953846584,
                                     1.6776961308198861, 1.555131926310589, 1.1806640179159833, 3.052979898534809,
                                     5.059864773115917, 1.1294288761955027, 2.6979674092930193, 9.82656489928625,
                                     5.501267559105623, 1.2351970327509707, 1.6078922552697488, 1.582561168396548,
                                     2.391340419810505, 2.660167843760268, 1.9569767827507456, 1.2776674104800188,
                                     1.6183122532590568, 4.552555359280379, 4.214077308248084, 1.7815460978157334,
                                     3.5980732014726056, 1.690396198372471, 4.671077054310741, 1.2339939189715687,
                                     2.7618415144011874, 2.86744016658739])

        self._ref_cont = UnbinnedContainer(self._ref_data)

        self.unbinned_fit = UnbinnedFit(data=self._ref_cont,
                                        model_density_function=self.data_model_density,
                                        minimizer=DEFAULT_TEST_MINIMIZER)

        self._ref_parameter_value_estimates = [2.12814778, 0.10562081]
        self._ref_parameter_value_estimates_2 = [1.64499273, 0.11695081]
        self._ref_model = self.data_model_density(self._ref_data, *self._ref_parameter_value_estimates)

    def test_before_fit_compare_parameter_values(self):
        self.assertTrue(
            np.allclose(
                self.unbinned_fit.parameter_values,
                self._ref_params
            )
        )

    def test_do_fit_compare_parameter_values(self):
        self.unbinned_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self.unbinned_fit.parameter_values,
                self._ref_parameter_value_estimates,
                rtol=1e-3
            )
        )

    def test_update_cost_function_on_parameter_change(self):
        self.unbinned_fit.set_all_parameter_values(self._ref_parameter_value_estimates)
        self.assertEqual(
            self.unbinned_fit.cost_function_value,
            self.unbinned_fit._cost_function(self._ref_model),
        )

    def test_report_before_fit(self):
        _buffer = six.StringIO()
        self.unbinned_fit.report(output_stream=_buffer)
        self.assertNotEquals(_buffer.getvalue(), "")

    def test_report_after_fit(self):
        _buffer = six.StringIO()
        self.unbinned_fit.do_fit()
        self.unbinned_fit.report(output_stream=_buffer)
        self.assertNotEquals(_buffer.getvalue(), "")

    def test_nexus_update_on_data_change(self):
        self.unbinned_fit.data = self._ref_data_2
        self.unbinned_fit.do_fit()
        print(self.unbinned_fit.parameter_values)
        self.assertTrue(
            np.allclose(
                self.unbinned_fit.parameter_values,
                self._ref_parameter_value_estimates_2,
                rtol=1e-3
            )
        )
