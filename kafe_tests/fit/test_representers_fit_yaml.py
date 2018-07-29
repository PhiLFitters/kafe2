import unittest
import numpy as np
from six import StringIO

from kafe.fit.representation import FitYamlWriter, FitYamlReader
from kafe.fit.io.handle import IOStreamHandle
from kafe.fit.xy import XYFit

TEST_FIT_XY="""
fit:
    type: xy
    dataset:
        type: xy
        x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        y_data: [ -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
                9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
        y_errors:
            - correlation_coefficient: 0.0
              error_value: 0.1
              name: test_y_error
              relative: false
              type: simple
    parametric_model:
        type: xy
        x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        model_function:
            type: xy
            python_code: |
                def linear_model(x, a, b):
                    return a * x + b
        model_parameters: [1.0, 1.0]
"""

class TestXYFitYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(x, a, b):
        return a * x + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_parameters = np.array([2.0, -1.0])
        self._test_parameters_default = np.array([1.0, 1.0])
        self._test_parameters_do_fit = np.array([2.0117809129092095, -1.090410559090481])
        self._test_y = [ -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
                        9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
        self._test_y_default = self.linear_model(self._test_x, *self._test_parameters_default)
        self._test_y_do_fit = [ -1.09041056, 0.92137035, 2.93315127, 4.94493218, 6.95671309,
                               8.96849401, 10.98027492, 12.99205583, 15.00383674, 17.01561766]
        
        self._fit = XYFit(
            xy_data=[self._test_x, self._test_y],
            model_function=TestXYFitYamlRepresenter.linear_model
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_FIT_XY))
        
        self._roundtrip_streamreader = FitYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = FitYamlWriter(self._fit, self._roundtrip_stringstream)
        self._testfile_streamreader = FitYamlReader(self._testfile_stringstream)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, XYFit))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default,
                _read_fit.y_model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.y_model
            )
        )

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, XYFit))

        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default,
                _read_fit.y_model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.y_model
            )
        )
