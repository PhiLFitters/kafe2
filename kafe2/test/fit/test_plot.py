import unittest2 as unittest
import sys
from kafe2 import XYFit, Plot


# FIXME: Use different matplotlib backend instead of skipping the tests
@unittest.skipIf(sys.version_info[0] < 3, "Require Python3 for plotting on TravisCI servers.")
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

    def test_warning_no_fit_performed(self):
        _fit = XYFit(xy_data=self._ref_data)
        _fit.add_error('y', 0.1)
        _plot = Plot(fit_objects=_fit)
        with self.assertWarns(Warning) as w:
            _plot.plot()
        self.assertIn("Did you forget to run fit.do_fit()?", str(w.warning))

    def test_plot_with_asymmetric_errors(self):
        self.plot.plot(asymmetric_parameter_errors=True)
