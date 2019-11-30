import numpy as np
import six
import unittest2 as unittest

from kafe2.core.minimizers import AVAILABLE_MINIMIZERS
from kafe2.core.fitters.nexus import Nexus, Parameter, Function
from kafe2.core.fitters.nexus_fitter import NexusFitter, NexusFitterException


class AbstractTestNexusFitter(object):

    MINIMIZER = None

    @staticmethod
    def slsq(x, y, x_1, y_1, x_2, y_2):
        return (
            (x - x_1)**2 + (y - y_1)**2 +
            (x - x_2)**2 + (y - y_2)**2
        )

    def setUp(self):
        self.nexus = Nexus()

        self._ref_xy_1 = (3, 9)
        self._ref_xy_2 = (7, 11)
        self._ref_xy = tuple(
            0.5*(np.array(self._ref_xy_1) + np.array(self._ref_xy_2))
        )
        self._startval_x = 300
        self._startval_y = -20

        par_x_1, par_y_1 = (
            self.nexus.add(
                Parameter(self._ref_xy_1[0], name='x_1')
            ),
            self.nexus.add(
                Parameter(self._ref_xy_1[1], name='y_1')
            )
        )
        par_x_2, par_y_2 = (
            self.nexus.add(
                Parameter(self._ref_xy_2[0], name='x_2')
            ),
            self.nexus.add(
                Parameter(self._ref_xy_2[1], name='y_2')
            )
        )

        par_x, par_y = (
            self.nexus.add(
                Parameter(self._startval_x, name='x')
            ),
            self.nexus.add(
                Parameter(self._startval_y, name='y')
            )
        )

        self.nexus.add_function(self.slsq)

        self.fitter = NexusFitter(
            self.nexus,
            parameters_to_fit=('x', 'y'),
            parameter_to_minimize='slsq',
            minimizer=self.MINIMIZER
        )

    def _assert_fit_results(self, xy_val=None):

        xy_val = xy_val or self._ref_xy

        for _par_name, _val in zip(('x', 'y'), xy_val):
            self.assertAlmostEqual(
                self.fitter.get_fit_parameter_values()[_par_name],
                _val,
                places=3
            )

        _slsq_args = tuple(xy_val) + self._ref_xy_1 + self._ref_xy_2
        self.assertAlmostEqual(
            self.fitter.parameter_to_minimize_value,
            self.slsq(*_slsq_args),
            places=2
        )

    # -- init

    def test_init_inexistent_fit_parameters_raise(self):
        with self.assertRaises(NexusFitterException):
            NexusFitter(
                self.nexus,
                parameters_to_fit=('bogus_parameter',),
                parameter_to_minimize='slsq',
            )

    def test_init_inexistent_min_parameter_raise(self):
        with self.assertRaises(NexusFitterException):
            NexusFitter(
                self.nexus,
                parameters_to_fit=('x', 'y'),
                parameter_to_minimize='bogus_parameter',
            )

    # -- basic interface

    def test_simple_properties(self):
        self.assertEqual(
            self.fitter.parameters_to_fit,
            ('x', 'y')
        )
        self.assertEqual(
            self.fitter.parameter_to_minimize,
            'slsq'
        )
        self.assertEqual(
            self.fitter.get_fit_parameter_values(),
            {
                'x': self._startval_x,
                'y': self._startval_y,
            }
        )
        self.assertEqual(
            self.fitter.n_fit_par,
            2
        )
        self._assert_fit_results(xy_val=[self._startval_x, self._startval_y])

    def test_set_all_fit_parameter_values(self):
        self.fitter.set_all_fit_parameter_values([7, 2])
        self._assert_fit_results(xy_val=[7, 2])

    def test_set_all_fit_parameter_values_length_mismatch_raise(self):
        with self.assertRaises(NexusFitterException):
            self.fitter.set_all_fit_parameter_values([7, 2, 30])
        with self.assertRaises(NexusFitterException):
            self.fitter.set_all_fit_parameter_values([2])

    def test_set_fit_parameter_values(self):
        self.fitter.set_fit_parameter_values(x=7, y=2)
        self._assert_fit_results(xy_val=[7, 2])

    def test_set_fit_parameter_values_unknown_raise(self):
        with self.assertRaises(NexusFitterException):
            self.fitter.set_fit_parameter_values(bogus_parameter=7)

    def test_do_fit(self):
        self.fitter.do_fit()
        self._assert_fit_results()  # nominal fit results

    def test_state_is_from_minimizer(self):
        self.assertEqual(self.fitter.state_is_from_minimizer, False)
        self.fitter.do_fit()
        self.assertEqual(self.fitter.state_is_from_minimizer, True)
        self.fitter.set_fit_parameter_values(x=7)
        self.assertEqual(self.fitter.state_is_from_minimizer, False)

    # -- parameter errors

    def test_fit_parameter_errors(self):
        self.fitter.do_fit()
        self.assertAlmostEqual(
            self.fitter.fit_parameter_errors[0],
            1.0/np.sqrt(2)
        )
        self.assertAlmostEqual(
            self.fitter.fit_parameter_errors[1],
            1.0/np.sqrt(2)
        )

    def test_fit_parameter_cov_mat(self):
        self.fitter.do_fit()
        self.assertAlmostEqual(
            self.fitter.fit_parameter_cov_mat[0, 0],
            0.5
        )

    def test_fit_parameter_cor_mat(self):
        self.fitter.do_fit()
        self.assertAlmostEqual(
            self.fitter.fit_parameter_cor_mat[0, 0],
            1.0
        )

    def test_asymmetric_fit_parameter_errors(self):
        self.fitter.do_fit()
        self.assertAlmostEqual(
            self.fitter.asymmetric_fit_parameter_errors[0][0],
            -1.0/np.sqrt(2),
            places=4
        )
        self.assertAlmostEqual(
            self.fitter.asymmetric_fit_parameter_errors[0][1],
            1.0/np.sqrt(2),
            places=4
        )

    def test_asymmetric_fit_parameter_errors_if_calculated(self):
        self.fitter.do_fit()
        self.assertIs(
            self.fitter.asymmetric_fit_parameter_errors_if_calculated,
            None
        )

    # -- fixing & limiting parameters

    def test_fix_parameter(self):
        self.fitter.fix_parameter('x')
        self.fitter.do_fit()
        self._assert_fit_results(xy_val=[self._startval_x, self._ref_xy[1]])

    def test_fix_parameter_to_value(self):
        self.fitter.fix_parameter('x', 200)
        self.fitter.do_fit()
        self._assert_fit_results(xy_val=[200, self._ref_xy[1]])

    def test_release_parameter(self):
        self.fitter.fix_parameter('x')
        self.fitter.do_fit()
        self.fitter.release_parameter('x')
        self.fitter.do_fit()
        self._assert_fit_results()  # nominal fit results

    def test_limit_parameter(self):
        self.fitter.limit_parameter('x', (6, 10))
        self.fitter.do_fit()
        self._assert_fit_results(xy_val=[6, self._ref_xy[1]])

    def test_unlimit_parameter(self):
        self.fitter.limit_parameter('x', (6, 10))
        self.fitter.do_fit()
        self.fitter.unlimit_parameter('x')
        self.fitter.do_fit()
        self._assert_fit_results()  # nominal fit results


    # -- profiles & contours

    def test_profile_without_do_fit(self):
        with self.assertRaises(NexusFitterException):
            self.fitter.profile('x')

    def test_profile(self):
        self.fitter.do_fit()
        p = self.fitter.profile('x', bins=20, bound=2)
        self.assertEqual(
            p.shape,
            (2, 20)
        )
        self.assertAlmostEqual(
            p[0][-1] - p[0][0],
            4 * self.fitter.fit_parameter_errors[0]
        )

    def test_contour_without_do_fit(self):
        with self.assertRaises(NexusFitterException):
            self.fitter.contour('x', 'y')

    def test_contour(self):
        self.fitter.do_fit()
        c = self.fitter.contour('x', 'y')
        # TODO: test result?


if 'scipy' in AVAILABLE_MINIMIZERS:
    class TestNexusFitterScipy(AbstractTestNexusFitter, unittest.TestCase):
        MINIMIZER = 'scipy'

if 'iminuit' in AVAILABLE_MINIMIZERS:
    class TestNexusFitterIMinuit(AbstractTestNexusFitter, unittest.TestCase):
        MINIMIZER = 'iminuit'
