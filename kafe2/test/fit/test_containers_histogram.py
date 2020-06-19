import unittest2 as unittest
import numpy as np
import scipy.stats as stats

from kafe2.fit import HistContainer, HistParametricModel
from kafe2.fit._base import DataContainerException
from kafe2.fit.histogram.container import HistContainerException
from kafe2.fit.histogram.model import HistParametricModelException,\
    HistModelFunction
from kafe2.fit.util.function_library import linear_model, quadratic_model


class TestDatastoreHistogram(unittest.TestCase):

    def setUp(self):
        self._ref_entries = [-9999., -8279., 3.3, 5.5, 2.2, 8.5, 10., 10.2, 10000., 1e7]
        self._ref_n_bins_auto = 25
        self._ref_n_bins_manual = 10
        self._ref_n_bin_range = (0., 10.)

        self._ref_bin_edges_manual_equalspacing = np.linspace(0, 10, self._ref_n_bins_manual + 1)
                                                    # [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10]
        self._ref_data_manual_equalspacing = np.array([  0,  0,  1,  1,  0,  1,  0,  0,  1,  0])
        self._ref_bin_heights_manual =       np.array([  5,  4, 10,  3,  0, 15,  7,  4, 20,  1])

        self._ref_bin_edges_manual_variablespacing =   [0 , 2 , 3 , 3.1 , 3.2 , 3.3 , 3.4 , 7 , 8.5 , 9 , 10]
        self._ref_data_manual_variablespacing = np.array([  0,  1,  0,    0,    0,    1,    1,  0,    1,  0])

        self._ref_data_auto = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

        # test rebinning functionality
        self._probe_bin_edges_variablespacing_withedges =    [0,  2, 3, 3.1, 3.2, 3.3, 3.4, 7, 8.5, 9, 10]    # OK
        self._probe_bin_edges_variablespacing_noedges =      [    2, 3, 3.1, 3.2, 3.3, 3.4, 7, 8.5, 9]        # OK
        self._probe_bin_edges_variablespacing_wrongedges1 =  [0,  2, 3, 3.1, 3.2, 3.3, 3.4, 7, 8.5, 9, 12.3]  # fail
        self._probe_bin_edges_variablespacing_wrongedges2 =  [-9, 2, 3, 3.1, 3.2, 3.3, 3.4, 7, 8.5, 9, 10]   # fail
        self._probe_bin_edges_variablespacing_wrongedges3 =  [-3, 2, 3, 3.1, 3.2, 3.3, 3.4, 7, 8.5, 9, 22]   # fail
        self._probe_bin_edges_variablespacing_wrongnumber =  [0,  2, 3, 3.1, 3.2, 3.3, 3.4, 7, 8.5, 10]       # fail
        self._probe_bin_edges_variablespacing_unsorted =     [0,  2, 3, 8.5, 3.2, 3.3, 3.4, 7, 3.1, 9, 10]    # fail


        self.hist_cont_binedges_auto = HistContainer(self._ref_n_bins_auto, self._ref_n_bin_range, bin_edges=None)
        self.hist_cont_binedges_manual_equal = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range, bin_edges=self._ref_bin_edges_manual_equalspacing)
        self.hist_cont_binedges_manual_variable = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range, bin_edges=self._ref_bin_edges_manual_variablespacing)


    def test_property_size(self):
        self.assertEqual(self.hist_cont_binedges_auto.size, self._ref_n_bins_auto)
        self.assertEqual(self.hist_cont_binedges_manual_equal.size, self._ref_n_bins_manual)
        self.assertEqual(self.hist_cont_binedges_manual_variable.size, self._ref_n_bins_manual)

    def test_property_low(self):
        self.assertEqual(self.hist_cont_binedges_auto.low, self._ref_n_bin_range[0])
        self.assertEqual(self.hist_cont_binedges_manual_equal.low, self._ref_n_bin_range[0])
        self.assertEqual(self.hist_cont_binedges_manual_variable.low, self._ref_n_bin_range[0])

    def test_property_high(self):
        self.assertEqual(self.hist_cont_binedges_auto.high, self._ref_n_bin_range[1])
        self.assertEqual(self.hist_cont_binedges_manual_equal.high, self._ref_n_bin_range[1])
        self.assertEqual(self.hist_cont_binedges_manual_variable.high, self._ref_n_bin_range[1])

    def test_fill_empty_binedges_auto_compare_data(self):
        self.hist_cont_binedges_auto.fill(self._ref_entries)
        self.assertTrue(
            np.allclose(self.hist_cont_binedges_auto.data, self._ref_data_auto)
        )

    def test_fill_empty_binedges_manual_equal_compare_data(self):
        self.hist_cont_binedges_manual_equal.fill(self._ref_entries)
        self.assertTrue(
            np.allclose(self.hist_cont_binedges_manual_equal.data, self._ref_data_manual_equalspacing)
        )

    def test_fill_empty_binedges_manual_variable_compare_data(self):
        self.hist_cont_binedges_manual_variable.fill(self._ref_entries)
        self.assertTrue(
            np.allclose(self.hist_cont_binedges_manual_variable.data, self._ref_data_manual_variablespacing)
        )

    def test_fill_empty_binedges_auto_rebin_manual_equal_compare_data(self):
        self.hist_cont_binedges_auto.fill(self._ref_entries)
        self.hist_cont_binedges_auto.rebin(self._ref_bin_edges_manual_equalspacing)
        self.assertTrue(
            np.allclose(self.hist_cont_binedges_auto.data, self._ref_data_manual_equalspacing)
        )

    def test_fill_empty_binedges_auto_rebin_manual_variable_compare_data(self):
        self.hist_cont_binedges_auto.fill(self._ref_entries)
        self.hist_cont_binedges_auto.rebin(self._ref_bin_edges_manual_variablespacing)
        self.assertTrue(
            np.allclose(self.hist_cont_binedges_auto.data, self._ref_data_manual_variablespacing)
        )

    def test_manual_bin_height(self):
        self.hist_cont_binedges_manual_equal.set_bins(self._ref_bin_heights_manual)
        self.assertTrue(np.alltrue(self.hist_cont_binedges_manual_equal.data == self._ref_bin_heights_manual))
        self.assertTrue(self.hist_cont_binedges_manual_equal._manual_heights)

    def test_construct_bin_edges_variablespacing_withedges(self):
        _hc = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range, bin_edges=self._probe_bin_edges_variablespacing_withedges)

    def test_construct_bin_edges_variablespacing_noedges(self):
        _hc = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range, bin_edges=self._probe_bin_edges_variablespacing_noedges)

    def test_raise_construct_bin_edges_variablespacing_wrongedges1(self):
        with self.assertRaises(HistContainerException):
            _hc = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range,
                                bin_edges=self._probe_bin_edges_variablespacing_wrongedges1)

    def test_raise_construct_bin_edges_variablespacing_wrongedges2(self):
        with self.assertRaises(HistContainerException):
            _hc = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range,
                                bin_edges=self._probe_bin_edges_variablespacing_wrongedges2)

    def test_raise_construct_bin_edges_variablespacing_wrongedges3(self):
        with self.assertRaises(HistContainerException):
            _hc = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range,
                                bin_edges=self._probe_bin_edges_variablespacing_wrongedges3)

    def test_raise_construct_bin_edges_variablespacing_wrongnumber(self):
        with self.assertRaises(HistContainerException):
            _hc = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range,
                                bin_edges=self._probe_bin_edges_variablespacing_wrongnumber)

    def test_raise_construct_bin_edges_variablespacing_unsorted(self):
        with self.assertRaises(HistContainerException):
            _hc = HistContainer(self._ref_n_bins_manual, self._ref_n_bin_range,
                                bin_edges=self._probe_bin_edges_variablespacing_unsorted)

    def test_raise_add_same_error_name_twice(self):
        self.hist_cont_binedges_auto.add_error(0.1,
                                               name="MyNewError",
                                               correlation=0, relative=False)
        with self.assertRaises(DataContainerException):
            self.hist_cont_binedges_auto.add_error(0.1,
                                                   name="MyNewError",
                                                   correlation=0, relative=False)

    def test_raise_get_inexistent_error(self):
        with self.assertRaises(DataContainerException):
            self.hist_cont_binedges_auto.get_error("MyInexistentError")

    def test_raise_manual_bin_heights_rebin(self):
        self.hist_cont_binedges_manual_equal.set_bins(self._ref_bin_heights_manual)
        with self.assertRaises(DataContainerException):
            self.hist_cont_binedges_manual_equal.rebin(self._ref_bin_edges_manual_variablespacing)


class TestDatastoreHistParametricModel(unittest.TestCase):
    @staticmethod
    def _ref_model_func(x, mu, sigma):
        return stats.norm(mu, sigma).pdf(x)

    @staticmethod
    def _ref_model_func_antider(x, mu, sigma):
        return stats.norm(mu, sigma).cdf(x)

    def setUp(self):
        self._ref_pm_support = np.linspace(-5, 5, 11)

        self._ref_n_bins = 11
        self._ref_n_bin_range = (-3, 25)
        self._ref_bin_edges = np.linspace(self._ref_n_bin_range[0], self._ref_n_bin_range[1], self._ref_n_bins+1)

        self._ref_params = (14., 3.)
        self._ref_data = (self._ref_model_func_antider(self._ref_bin_edges[1:], *self._ref_params) -
                          self._ref_model_func_antider(self._ref_bin_edges[:-1], *self._ref_params))

        self.hist_param_model_numerical = HistParametricModel(
            n_bins=self._ref_n_bins,
            bin_range=self._ref_n_bin_range,
            model_density_func=HistModelFunction(self._ref_model_func), 
            model_parameters=self._ref_params,
            bin_edges=None, 
            bin_evaluation="numerical")

        self.hist_param_model_with_antider = HistParametricModel(
            n_bins=self._ref_n_bins,
            bin_range=self._ref_n_bin_range,
            model_density_func=HistModelFunction(self._ref_model_func),
            model_parameters=self._ref_params,
            bin_edges=None, bin_evaluation=self._ref_model_func_antider)

        self.hist_param_model_np_vectorize = HistParametricModel(
            n_bins=self._ref_n_bins, bin_range=self._ref_n_bin_range,
            model_density_func=self._ref_model_func,
            model_parameters=self._ref_params, bin_edges=None,
            bin_evaluation=np.vectorize(self._ref_model_func_antider)
        )

        self._test_params = (20., 5.)
        self._ref_test_data = (self._ref_model_func_antider(self._ref_bin_edges[1:], *self._test_params) -
                          self._ref_model_func_antider(self._ref_bin_edges[:-1], *self._test_params))

    def test_compare_hist_model_no_antider_ref_data(self):
        self.assertTrue(np.allclose(self.hist_param_model_numerical.data, self._ref_data))

    def test_compare_hist_model_with_antider_ref_data(self):
        self.assertTrue(np.allclose(self.hist_param_model_with_antider.data, self._ref_data))

    def test_compare_hist_model_np_vectorize_ref_data(self):
        self.assertTrue(np.allclose(self.hist_param_model_np_vectorize.data, self._ref_data))

    def test_change_parameters_test_data(self):
        self.hist_param_model_numerical.parameters = self._test_params
        self.assertTrue(np.allclose(self.hist_param_model_numerical.data, self._ref_test_data))

    def test_raise_set_data(self):
        with self.assertRaises(HistParametricModelException):
            self.hist_param_model_numerical.data = self._ref_test_data

    def test_raise_fill(self):
        with self.assertRaises(HistParametricModelException):
            self.hist_param_model_numerical.fill([-1, 2, 700])

    def test_raise_unknown_bin_evaluation_raise(self):
        with self.assertRaises(ValueError):
            HistParametricModel(n_bins=10, bin_range=(-5, 5), bin_evaluation="magical")
        with self.assertRaises(ValueError):
            HistParametricModel(n_bins=10, bin_range=(-5, 5), bin_evaluation=5)


class TestQuadrature(unittest.TestCase):

    @staticmethod
    def _get_models(bin_evaluation):

        def _sinus_model(x, a):
            return np.sin(x) + a

        return (
            HistParametricModel(
                n_bins=10, bin_range=(0, 10), model_density_func=linear_model,
                model_parameters=[1.0, 1.0], bin_evaluation=bin_evaluation
            ), HistParametricModel(
                n_bins=10, bin_range=(0, 10), model_density_func=quadratic_model,
                model_parameters=[-0.1, 1.0, 1.0], bin_evaluation=bin_evaluation
            ), HistParametricModel(
                n_bins=1000, bin_range=(0, 10), model_density_func=quadratic_model,
                model_parameters=[-0.1, 1.0, 1.0], bin_evaluation=bin_evaluation
            ), HistParametricModel(
                n_bins=1000, bin_range=(0, 2 * np.pi), model_density_func=_sinus_model,
                model_parameters=[2.0], bin_evaluation=bin_evaluation
            )
        )

    def setUp(self):
        (
            self._linear_model_numerical,
            self._quadratic_model_numerical,
            self._quadratic_model_limit_numerical,
            self._sinus_model_limit_numerical
        ) = TestQuadrature._get_models("numerical")

    def test_rectangle(self):
        (
            _linear_model_rectangle,
            _quadratic_model_rectangle,
            _quadratic_model_limit_rectangle,
            _sinus_model_limit_rectangle
        ) = TestQuadrature._get_models("rectangle")
        self.assertTrue(np.allclose(
            self._linear_model_numerical.data, _linear_model_rectangle.data
        ))
        self.assertTrue(np.allclose(
            self._quadratic_model_limit_numerical.data, _quadratic_model_limit_rectangle.data
        ))
        self.assertTrue(np.allclose(
            self._sinus_model_limit_numerical.data, _sinus_model_limit_rectangle.data
        ))

    def test_trapezoid(self):
        (
            _linear_model_trapezoid,
            _quadratic_model_trapezoid,
            _quadratic_model_limit_trapezoid,
            _sinus_model_limit_rectangle
        ) = TestQuadrature._get_models("trapezoid")
        self.assertTrue(np.allclose(
            self._linear_model_numerical.data, _linear_model_trapezoid.data
        ))
        self.assertTrue(np.allclose(
            self._quadratic_model_limit_numerical.data, _quadratic_model_limit_trapezoid.data
        ))
        self.assertTrue(np.allclose(
            self._sinus_model_limit_numerical.data, _sinus_model_limit_rectangle.data
        ))

    def test_simpson(self):
        (
            _linear_model_simpson,
            _quadratic_model_simpson,
            _quadratic_model_limit_simpson,
            _sinus_model_limit_simpson
        ) = TestQuadrature._get_models("simpson")
        self.assertTrue(np.allclose(
            self._linear_model_numerical.data, _linear_model_simpson.data
        ))
        self.assertTrue(np.allclose(
            self._quadratic_model_numerical.data, _quadratic_model_simpson.data
        ))
        self.assertTrue(np.allclose(
            self._quadratic_model_limit_numerical.data, _quadratic_model_limit_simpson.data
        ))
        self.assertTrue(np.allclose(
            self._sinus_model_limit_numerical.data, _sinus_model_limit_simpson.data
        ))
