import unittest
import numpy as np
import six

from scipy import stats

from kafe2.config import kc
from kafe2.fit import UnbinnedContainer, UnbinnedFit
from kafe2.fit.unbinned.fit import UnbinnedFitException


CONFIG_PARAMETER_DEFAULT_VALUE = kc('core', 'default_initial_parameter_value')


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

        self._ref_data = np.array([7.42, 3.773, 5.968, 4.924, 1.468, 4.664, 1.745, 2.144, 3.836, 3.132, 1.568, 2.352,
                                   2.132, 9.381, 1.484, 1.181, 5.004, 3.06, 4.582, 2.076, 1.88, 1.337, 3.092, 2.265,
                                   1.208, 2.753, 4.457, 3.499, 8.192, 5.101, 1.572, 5.152, 4.181, 3.52, 1.344, 10.29,
                                   1.152, 2.348, 2.228, 2.172, 7.448, 1.108, 4.344, 2.042, 5.088, 1.02, 1.051, 1.987,
                                   1.935, 3.773, 4.092, 1.628, 1.688, 4.502, 4.687, 6.755, 2.56, 1.208, 2.649, 1.012,
                                   1.73, 2.164, 1.728, 4.646, 2.916, 1.101, 2.54, 1.02, 1.176, 4.716, 9.671, 1.692,
                                   9.292, 10.72, 2.164, 2.084, 2.616, 1.584, 5.236, 3.663, 3.624, 1.051, 1.544, 1.496,
                                   1.883, 1.92, 5.968, 5.89, 2.896, 2.76, 1.475, 2.644, 3.6, 5.324, 8.361, 3.052, 7.703,
                                   3.83, 1.444, 1.343])

        self._ref_cont = UnbinnedContainer(self._ref_data)

        self.unbinned_fit = UnbinnedFit(data=self._ref_cont, model_density_function=self.data_model_density)

        self._ref_parameter_value_estimates = [2.12814778, 0.10562081]
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