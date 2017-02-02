import unittest
import numpy as np

from kafe.fit.datastore.histogram import HistContainer, HistContainerException


class TestDatastoreHistogram(unittest.TestCase):

    def setUp(self):
        self._ref_entries = [-9999., -8279., 3.3, 5.5, 2.2, 8.5, 10., 10.2, 10000., 1e7]
        self._ref_n_bins_auto = 25
        self._ref_n_bins_manual = 10
        self._ref_n_bin_range = (0., 10.)

        self._ref_bin_edges_manual_equalspacing = np.linspace(0, 10, self._ref_n_bins_manual + 1)
                                                    # [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10]
        self._ref_data_manual_equalspacing = np.array([  0,  0,  1,  1,  0,  1,  0,  0,  1,  0])

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

