import unittest2 as unittest
import numpy as np

from kafe2.fit import CustomFit

from kafe2.test.fit.test_fit import AbstractTestFit

class TestCustomFitWithSimpleYErrors(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    @staticmethod
    def chi2(a=1.5, b=-0.5):
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        y_data = np.array([
            -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206, 9.15059627, 10.9665515,
            13.06741151, 14.95081026, 16.94404467
        ])
        y_err = 0.1
        y_model = a * x_data + b
        cost = np.sum(np.square((y_model - y_data) / y_err))
        cost += 2 * x_data.shape[0] * np.log(y_err)
        return cost

    def setUp(self):
        self._n_points = 10

        # reference initial values
        self._ref_initial_pars = np.array([1.5, -0.5])

        # pre-fit cost value
        self._ref_initial_cost = self.chi2(*self._ref_initial_pars)

        self._nominal_fit_result_pars = np.array([2.0117809, -1.09041046])

        self._nominal_fit_result_cost = self.chi2(*self._nominal_fit_result_pars)

        # helper dict with all reference property values
        self._ref_prop_dict = dict(
            did_fit=False,
            model_count=0,

            parameter_values=self._ref_initial_pars,
            parameter_names=('a', 'b'),
            cost_function_value=self._ref_initial_cost,

            data=None,
            data_error=None,
            data_cov_mat=None,
            data_cov_mat_inverse=None,
            data_cor_mat=None,

            model=None,
            model_error=None,
            model_cov_mat=None,
            model_cov_mat_inverse=None,
            model_cor_mat=None,

            total_error=None,
            total_cov_mat=None,
            total_cov_mat_inverse=None,
            total_cor_mat=None,
        )

    def _get_fit(self):
        '''convenience'''
        _fit = CustomFit(
            cost_function=self.chi2,
            minimizer=self.MINIMIZER
        )
        return _fit

    def _get_test_fits(self):
        return {
            'default': self._get_fit(),
        }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            self._ref_prop_dict,
            fit_names=['default'],
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars,
                did_fit=True,
                cost_function_value=self._nominal_fit_result_cost,
            ),
            fit_names=['default'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-2
        )
