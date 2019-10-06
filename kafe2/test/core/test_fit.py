import unittest2 as unittest

import numpy as np

from kafe2.core.fitters import Nexus, NexusFitter, NexusFitterException


# class TestNexusFitter(unittest.TestCase):
#
#     def setUp(self):
#         self.ps = Nexus()
#         self.n_points = 10
#
#         self.x_support = np.arange(self.n_points)
#         self.y_true_errors = np.array([-0.14828884, -0.01484907, -1.07937053,
#                                        -0.16996879,  1.48169694, -0.24361865,
#                                         1.80168599, -0.80596291,  1.64451791,
#                                        -1.05440662])
#         self.y_measured = np.arange(self.n_points) + self.y_true_errors
#         self.y_error_model = np.ones(self.n_points) * 1.0
#
#         self.ps.new(x_support=self.x_support)
#         self.ps.new(y_measured=self.y_measured)
#         self.ps.new(y_errors=self.y_error_model)
#
#         self.slope = 1.0
#         self.y_intercept = 0.0
#         self.chi2_initial = 11.182777169656681
#         self.ps.new(slope=self.slope)
#         self.ps.new(y_intercept=self.y_intercept)
#
#         def y_theory(x_support, slope, y_intercept):
#             return slope * x_support + y_intercept
#
#         def chi2(y_measured, y_theory, y_errors):
#             _p = y_measured - y_theory
#             return np.sum((_p / y_errors) ** 2)
#
#         self.ps.new_function(y_theory)
#         self.ps.new_function(chi2)
#
#         self.sf = NexusFitter(self.ps, ('slope', 'y_intercept'), 'chi2')
#         self.sf_only_slope = NexusFitter(self.ps, ('slope',), 'chi2')
#
#         self.slope_after_fit = 1.0546496459587062
#         self.y_intercept_after_fit = -0.10477992306698165
#         self.chi2_after_fit = 10.737168784133402
#
#         self.slope_after_fit_only_slope = 1.0381054758557233
#         self.chi2_after_fit_only_slope = 10.768949646411523
#
#     def test_compare_par_values_initial(self):
#         self.assertDictEqual(self.sf.fit_parameter_values, dict(slope=self.slope, y_intercept=self.y_intercept))
#
#     def test_compare_par_values_after_fit(self):
#         self.sf.do_fit()
#         self.assertDictEqual(self.sf.fit_parameter_values, dict(slope=self.slope_after_fit, y_intercept=self.y_intercept_after_fit))
#
#     def test_compare_par_values_after_fit_only_slope(self):
#         self.sf_only_slope.do_fit()
#         self.assertDictEqual(self.sf_only_slope.fit_parameter_values, dict(slope=self.slope_after_fit_only_slope))
#
#     def test_compare_chi2_value_initial(self):
#         self.assertEqual(self.sf.parameter_to_minimize_value, self.chi2_initial)
#
#     def test_compare_chi2_value_after_fit(self):
#         self.sf.do_fit()
#         self.assertEqual(self.sf.parameter_to_minimize_value, self.chi2_after_fit)
#
#     def test_compare_chi2_value_after_fit_only_slope(self):
#         self.sf_only_slope.do_fit()
#         self.assertEqual(self.sf_only_slope.parameter_to_minimize_value, self.chi2_after_fit_only_slope)

class TestNexusFitterIMinuit(unittest.TestCase):

    def setUp(self):
        self.ps = Nexus()
        self.n_points = 10

        self.x_support = np.arange(self.n_points)
        self.y_true_errors = np.array([-0.14828884, -0.01484907, -1.07937053,
                                       -0.16996879,  1.48169694, -0.24361865,
                                        1.80168599, -0.80596291,  1.64451791,
                                       -1.05440662])
        self.y_measured = np.arange(self.n_points) + self.y_true_errors
        self.y_error_model = np.ones(self.n_points) * 1.0

        self.ps.new(x_support=self.x_support)
        self.ps.new(y_measured=self.y_measured)
        self.ps.new(y_errors=self.y_error_model)

        self.slope = 1.0
        self.y_intercept = 0.0
        self.chi2_initial = 11.182777169656681
        self.ps.new(slope=self.slope)
        self.ps.new(y_intercept=self.y_intercept)

        def y_theory(x_support, slope, y_intercept):
            return slope * x_support + y_intercept

        def chi2(y_measured, y_theory, y_errors):
            _p = y_measured - y_theory
            return np.sum((_p / y_errors) ** 2)

        self.ps.new_function(y_theory)
        self.ps.new_function(chi2)

        self.sf = NexusFitter(self.ps, ('slope', 'y_intercept'), 'chi2', minimizer=None)
        self.sf_only_slope = NexusFitter(self.ps, ('slope',), 'chi2', minimizer=None)

        self.slope_after_fit = 1.0546496459587062
        self.y_intercept_after_fit = -0.10477992306698165
        self.chi2_after_fit = 10.737168784133402

        self.slope_after_fit_only_slope = 1.0381054758557233
        self.chi2_after_fit_only_slope = 10.768949646411523

    def test_compare_par_values_initial(self):
        self.assertDictEqual(self.sf.fit_parameter_values, dict(slope=self.slope, y_intercept=self.y_intercept))

    def test_compare_par_values_after_fit(self):
        self.sf.do_fit()
        _ref_dict = dict(slope=self.slope_after_fit, y_intercept=self.y_intercept_after_fit)
        _chk_dict = self.sf.fit_parameter_values
        self.assertEqual(set(_ref_dict.keys()), set(_chk_dict.keys()))
        for _pn in _chk_dict:
            self.assertAlmostEqual(_chk_dict[_pn],
                                   _ref_dict[_pn],
                                   places=2)

    def test_compare_par_values_after_fit_only_slope(self):
        self.sf_only_slope.do_fit()
        _ref_dict = dict(slope=self.slope_after_fit_only_slope)
        _chk_dict = self.sf_only_slope.fit_parameter_values
        self.assertEqual(set(_ref_dict.keys()), set(_chk_dict.keys()))
        for _pn in _chk_dict:
            self.assertAlmostEqual(_chk_dict[_pn],
                                   _ref_dict[_pn],
                                   places=2)

    def test_compare_chi2_value_initial(self):
        self.assertAlmostEqual(self.sf.parameter_to_minimize_value, self.chi2_initial, places=4)

    def test_compare_chi2_value_after_fit(self):
        self.sf.do_fit()
        self.assertAlmostEqual(self.sf.parameter_to_minimize_value, self.chi2_after_fit, places=4)

    def test_compare_chi2_value_after_fit_only_slope(self):
        self.sf_only_slope.do_fit()
        self.assertAlmostEqual(self.sf_only_slope.parameter_to_minimize_value, self.chi2_after_fit_only_slope, places=4)

    def test_set_fit_parameters_by_name_value_dict_compare_parameter_to_minimize_value(self):
        self.sf.set_fit_parameter_values(slope=self.slope_after_fit,
                                         y_intercept=self.y_intercept_after_fit)
        self.assertEqual(
            self.chi2_after_fit,
            self.sf.parameter_to_minimize_value
        )

    def test_set_all_fit_parameters_compare_parameter_to_minimize_value(self):
        self.sf.set_all_fit_parameter_values((self.slope_after_fit, self.y_intercept_after_fit))

        self.assertEqual(
            self.chi2_after_fit,
            self.sf.parameter_to_minimize_value
        )

    def test_raise_set_fit_parameters_unknown_name(self):
        with self.assertRaises(NexusFitterException):
            self.sf.set_fit_parameter_values(bogus=0.2836)

    def test_raise_set_fit_parameters_more_values(self):
        with self.assertRaises(NexusFitterException):
            self.sf.set_all_fit_parameter_values((self.slope_after_fit, self.y_intercept_after_fit, 1.234))

    def test_raise_set_fit_parameters_less_values(self):
        with self.assertRaises(NexusFitterException):
            self.sf.set_all_fit_parameter_values((self.slope_after_fit,))
