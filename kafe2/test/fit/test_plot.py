import unittest2 as unittest
import sys
import numpy as np
from kafe2 import XYFit, Plot


class TestXYPlot(unittest.TestCase):
    def setUp(self):
        self._ref_data = [[1, 2, 3], [1, 2, 3]]
        self._ref_dataset_label = 'My Dataset'
        self._ref_x_label = '$U$ [V]'
        self._ref_y_label = '$I$ [A]'
        self._ref_model_label = 'My Model'
        self._ref_error_label = self._ref_model_label + r' $\pm 1\sigma$'

        self.fit = XYFit(xy_data=self._ref_data)
        self.fit.add_error('y', 0.1)
        self.fit.do_fit()

        self.fit.data_container.label = self._ref_dataset_label
        self.fit.data_container.axis_labels = (self._ref_x_label, self._ref_y_label)
        self.fit.model_label = self._ref_model_label

        self.plot = Plot(self.fit)

    def test_warning_no_fit_performed(self):
        _fit = XYFit(xy_data=self._ref_data)
        _fit.add_error('y', 0.1)
        _plot = Plot(fit_objects=_fit)
        with self.assertWarns(Warning) as w:
            _plot.plot()
        self.assertIn("Did you forget to run fit.do_fit()?", str(w.warning))

    def test_plot_with_asymmetric_errors(self):
        self.plot.plot(asymmetric_parameter_errors=True)

    def test_labels_after_plotting(self):
        self.plot.plot()
        self.assertEqual(self.plot.axes[0]['main'].get_xlabel(), self._ref_x_label)
        self.assertEqual(self.plot.axes[0]['main'].get_ylabel(), self._ref_y_label)
        _, labels = self.plot.axes[0]['main'].get_legend_handles_labels()
        self.assertIn(self._ref_dataset_label, labels)
        self.assertIn(self._ref_model_label, labels)
        self.assertIn(self._ref_error_label, labels)

    def test_remove_labels(self):
        self.fit.data_container.label = '__del__'
        self.fit.data_container.axis_labels = ('__del__', '__del__')
        self.fit.model_label = '__del__'
        self.plot.plot()
        _, labels = self.plot.axes[0]['main'].get_legend_handles_labels()
        self.assertEqual(self.plot.axes[0]['main'].get_xlabel(), '')
        self.assertEqual(self.plot.axes[0]['main'].get_ylabel(), '')
        self.assertTrue(len(labels) == 0)

    def test_label_setter(self):
        test_x, test_y = 'Test_x', 'Test_y'
        self.plot.x_label = test_x
        self.plot.y_label = test_y
        self.plot.plot()
        self.assertEqual(self.plot.axes[0]['main'].get_xlabel(), test_x)
        self.assertEqual(self.plot.axes[0]['main'].get_ylabel(), test_y)

    def test_x_log_scale(self):
        x_range_lin = self.plot._get_plot_adapters()[0].x_range
        self.plot.x_scale = 'log'
        x_range_log = self.plot._get_plot_adapters()[0].x_range
        # check if additional pad has adapted to new scale
        self.assertNotEqual(x_range_lin, x_range_log)
        self.plot.x_range = (1e-3, 4)
        self.assertItemsEqual(self.plot._get_plot_adapters()[0].model_line_x,
                              np.geomspace(1e-3, 4, 100))
        self.plot.plot()
        self.assertEqual(self.plot.axes[0]['main'].get_xscale(), 'log')

    def test_y_log_scale(self):
        self.plot.y_scale = 'log'
        self.plot.plot()
        self.assertEqual(self.plot.axes[0]['main'].get_yscale(), 'log')


class TestMultiPlot(unittest.TestCase):
    def setUp(self):
        self._ref_data1 = [[1, 2, 3], [1, 2, 3]]
        self._ref_data2 = [[1.1, 1.9, 2.5], [5, 3, 2]]
        self.fit1 = XYFit(xy_data=self._ref_data1)
        self.fit2 = XYFit(xy_data=self._ref_data2)
        self.fit1.add_error('y', 0.1)
        self.fit1.do_fit()
        self.fit2.add_error('y', 0.1)
        self.fit2.do_fit()

        self.plot = Plot([self.fit1, self.fit2])
        self.plot_sep = Plot([self.fit1, self.fit2], separate_figures=True)

    def test_label_setter_str(self):
        x_label, y_label = "123x", "Y_test"
        for _p in (self.plot, self.plot_sep):
            _p.x_label = x_label
            _p.y_label = y_label
            _p.plot()
        self.assertEqual(self.plot.axes[0]['main'].get_xlabel(), x_label)
        self.assertEqual(self.plot.axes[0]['main'].get_ylabel(), y_label)
        for _ax in self.plot_sep.axes:
            self.assertEqual(_ax['main'].get_xlabel(), x_label)
            self.assertEqual(_ax['main'].get_ylabel(), y_label)

    def test_label_setter_list(self):
        x_labels = ("X1", "$x_2$")
        y_labels = ("123", "123")
        for _p in (self.plot, self.plot_sep):
            _p.x_label = x_labels
            _p.y_label = y_labels
            _p.plot()
        # x label should be concatenation
        self.assertEqual(self.plot.axes[0]['main'].get_xlabel(), ', '.join(x_labels))
        # y label should not contain the same label twice
        self.assertEqual(self.plot.axes[0]['main'].get_ylabel(), y_labels[0])
        # each axis should have different labels for separate plots
        for _ax, _x_label, _y_label in zip(self.plot_sep.axes, x_labels, y_labels):
            self.assertEqual(_ax['main'].get_xlabel(), _x_label)
            self.assertEqual(_ax['main'].get_ylabel(), _y_label)

    def test_range_setter(self):
        x_ranges = ([1, 5], (0, 6.12))
        y_ranges = [(0, 3), [1, 6]]
        self.plot_sep.x_range = x_ranges
        self.plot_sep.y_range = y_ranges
        for _adapter, _xr, _yr in zip(self.plot_sep._get_plot_adapters(), x_ranges, y_ranges):
            self.assertEqual(_adapter.x_range, tuple(_xr))
            self.assertEqual(_adapter.y_range, tuple(_yr))

    def test_scale_setter(self):
        self.plot.x_scale = 'log'
        self.plot.y_scale = 'log'
        self.plot.plot()
        self.assertEqual(self.plot.axes[0]['main'].get_xscale(), 'log')
        self.assertEqual(self.plot.axes[0]['main'].get_yscale(), 'log')

        x_scales = ('linear', 'log')
        y_scales = ('log', 'linear')
        self.plot_sep.x_scale = x_scales
        self.plot_sep.y_scale = y_scales
        for _ax, _xs, _ys in zip(self.plot_sep.axes, x_scales, y_scales):
            self.assertEqual(_ax['main'].get_xscale(), _xs)
            self.assertEqual(_ax['main'].get_yscale(), _ys)
