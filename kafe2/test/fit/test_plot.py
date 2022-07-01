import unittest
import sys
import numpy as np
import os
from kafe2 import XYFit, Plot


class TestXYPlot(unittest.TestCase):
    def setUp(self):
        self._ref_data = [[1, 2, 3], [0.9, 2.1, 3.0]]
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
        self.assertTrue(
            np.all(self.plot._get_plot_adapters()[0].model_line_x == np.geomspace(1e-3, 4, 200)))
        self.plot.plot()
        self.assertEqual(self.plot.axes[0]['main'].get_xscale(), 'log')

    def test_y_log_scale(self):
        self.plot.y_scale = 'log'
        self.plot.plot()
        self.assertEqual(self.plot.axes[0]['main'].get_yscale(), 'log')

    def test_no_ratio_no_residual(self):
        self.plot.plot(ratio=False, residual=False)
        _current_axes = self.plot._current_axes

        with self.subTest():
            self.assertIn('main', _current_axes)
        with self.subTest():
            self.assertNotIn('ratio', _current_axes)
        with self.subTest():
            self.assertNotIn('residual', _current_axes)

    def test_ratio_no_residual(self):
        self.plot.plot(ratio=True, residual=False)
        _current_axes = self.plot._current_axes

        with self.subTest():
            self.assertIn('main', _current_axes)
        with self.subTest():
            self.assertIn('ratio', _current_axes)
            _ylim = _current_axes['ratio'].get_ylim()
            self.assertAlmostEqual(1 - _ylim[0], _ylim[1] - 1)
            _model = self.fit.y_model
            _data = self.fit.y_data
            _err = self.fit.total_error
            self.assertAlmostEqual(
                _ylim[1],
                1 + 1.05 * np.max((_err + np.abs(_model - _data)) / np.abs(_model)),
            )
        with self.subTest():
            self.assertNotIn('residual', _current_axes)

    def test_no_ratio_residual(self):
        self.plot.plot(ratio=False, residual=True)
        _current_axes = self.plot._current_axes

        with self.subTest():
            self.assertIn('main', _current_axes)
        with self.subTest():
            self.assertNotIn('ratio', _current_axes)
        with self.subTest():
            self.assertIn('residual', _current_axes)
            _ylim = _current_axes['residual'].get_ylim()
            self.assertAlmostEqual(-_ylim[0], _ylim[1])
            _model = self.fit.y_model
            _data = self.fit.y_data
            _err = self.fit.total_error
            self.assertAlmostEqual(
                _ylim[1],
                1.05 * np.max(_err + np.abs(_model - _data)),
            )

    def test_ratio_residual(self):
        with self.assertRaises(NotImplementedError):
            self.plot.plot(ratio=True, residual=True)

    def test_save(self):
        self.plot.plot()

        self.plot.save()
        self.assertTrue(os.path.exists("fit.png"))
        os.remove("fit.png")

        self.plot.save("my_fit.png")
        self.assertTrue(os.path.exists("my_fit.png"))
        os.remove("my_fit.png")

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

    def test_save(self):
        self.plot.plot()
        self.plot_sep.plot()

        self.plot.save()
        self.assertTrue(os.path.exists("fit.png"))
        os.remove("fit.png")

        self.plot.save("my_fit.png")
        self.assertTrue(os.path.exists("my_fit.png"))
        os.remove("my_fit.png")

        self.plot_sep.save()
        self.assertTrue(os.path.exists("fit_0.png"))
        os.remove("fit_0.png")
        self.assertTrue(os.path.exists("fit_1.png"))
        os.remove("fit_1.png")

        self.plot_sep.save(figures=0)
        self.assertTrue(os.path.exists("fit_0.png"))
        os.remove("fit_0.png")
        self.plot_sep.save(figures=1)
        self.assertTrue(os.path.exists("fit_1.png"))
        os.remove("fit_1.png")

        self.plot_sep.save("my_fit.png")
        self.assertTrue(os.path.exists("my_fit_0.png"))
        os.remove("my_fit_0.png")
        self.assertTrue(os.path.exists("my_fit_1.png"))
        os.remove("my_fit_1.png")

        self.plot_sep.save("one_fit.png", figures=0)
        self.assertTrue(os.path.exists("one_fit.png"))
        os.remove("one_fit.png")
        self.plot_sep.save("another_fit", figures=1)
        self.assertTrue(os.path.exists("another_fit.png"))
        os.remove("another_fit.png")

        self.plot_sep.save(["one_fit.png", "another_fit.png"])
        self.assertTrue(os.path.exists("one_fit.png"))
        os.remove("one_fit.png")
        self.assertTrue(os.path.exists("another_fit.png"))
        os.remove("another_fit.png")

    def test_save_raise(self):
        self.plot_sep.plot()
        with self.assertRaises(ValueError):
            self.plot_sep.save(fname=1)
        with self.assertRaises(ValueError):
            self.plot_sep.save(fname=["fit_0.png", "fit_1.png", "fit_2.png"])
