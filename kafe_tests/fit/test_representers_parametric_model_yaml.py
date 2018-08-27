import unittest
import numpy as np
from six import StringIO

from kafe.fit.histogram import HistModelFunction, HistModelDensityFunctionFormatter
from kafe.fit.indexed import IndexedModelFunction, IndexedModelFunctionFormatter
from kafe.fit.xy import XYModelFunction, XYModelFunctionFormatter
from kafe.fit.xy_multi import XYMultiModelFunction, XYMultiModelFunctionFormatter
from kafe.fit.representation import ParametricModelYamlWriter, ParametricModelYamlReader
from kafe.fit.xy.model import XYParametricModel
from kafe.fit.io.handle import IOStreamHandle

TEST_PARAMETRIC_MODEL_XY="""
type: xy
x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
model_function:
    type: xy
    python_code: |
        def linear_model(x, a, b):
            return a * x + b
model_parameters: [1.0, 1.0]
"""

TEST_PARAMETRIC_MODEL_XY_WITH_ERRORS="""
type: xy
x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
x_errors:
  - correlation_coefficient: 0.0
    error_value: 0.1
    name: test_x_error
    relative: false
    type: simple
model_function:
    type: xy
    python_code: |
        def linear_model(x, a, b):
            return a * x + b
model_parameters: [1.0, 1.0]
"""

class TestXYParametricModelYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(x, a, b):
        return a * x + b
   
    def setUp(self):
        self._test_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self._test_model_parameters = np.array([1.0, 1.0])
        self._test_parametric_model = XYParametricModel(
            self._test_x, 
            XYModelFunction(TestXYParametricModelYamlRepresenter.linear_model),
            self._test_model_parameters
        )

        self._test_parametric_model_with_errors = XYParametricModel(
            self._test_x, 
            XYModelFunction(TestXYParametricModelYamlRepresenter.linear_model),
            self._test_model_parameters
        )
        self._test_parametric_model_with_errors.add_simple_error(
            axis='x',
            err_val=0.1, 
            name='test_x_error', 
            correlation=0, 
            relative=False
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._roundtrip_stringstream_with_errors = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_XY))
        self._testfile_stringstream_with_errors = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_XY_WITH_ERRORS))
        
        self._roundtrip_streamreader = ParametricModelYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamreader_with_errors = ParametricModelYamlReader(self._roundtrip_stringstream_with_errors)
        self._roundtrip_streamwriter = ParametricModelYamlWriter(
            self._test_parametric_model, 
            self._roundtrip_stringstream
        )
        self._roundtrip_streamwriter_with_errors = ParametricModelYamlWriter(
            self._test_parametric_model_with_errors, 
            self._roundtrip_stringstream_with_errors
        )
        self._testfile_streamreader = ParametricModelYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_errors = ParametricModelYamlReader(self._testfile_stringstream_with_errors)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_write_to_roundtrip_stringstream_with_errors(self):
        self._roundtrip_streamwriter_with_errors.write()

    def test_read_from_testfile_stream(self):
        _read_parametric_model = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_parametric_model, XYParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        _read_parametric_model.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _read_parametric_model.y,
                self._test_parametric_model.y
            )
        )
        
    def test_read_from_testfile_stream_with_errors(self):
        _read_parametric_model = self._testfile_streamreader_with_errors.read()
        self.assertTrue(isinstance(_read_parametric_model, XYParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        _read_parametric_model.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _read_parametric_model.y,
                self._test_parametric_model_with_errors.y
            )
        )
        
    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter_with_errors.write()
        self._roundtrip_stringstream_with_errors.seek(0)  # return to beginning
        _read_parametric_model = self._roundtrip_streamreader_with_errors.read()
        self.assertTrue(isinstance(_read_parametric_model, XYParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        _read_parametric_model.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _read_parametric_model.y,
                self._test_parametric_model.y
            )
        )

        _given_error = self._test_parametric_model_with_errors.get_error('test_x_error')
        _read_error = _read_parametric_model.get_error('test_x_error')
        self.assertTrue(_given_error['axis'] == _read_error['axis'])
        self.assertTrue(_given_error['enabled'] == _read_error['enabled'])
        self.assertTrue(
            np.allclose(
                _given_error['err'].error,
                _read_error['err'].error
            )
        )
        self.assertTrue(_given_error['err'].corr_coeff == _read_error['err'].corr_coeff)
        
    def test_round_trip_with_stringstream_with_errors(self):
        self._roundtrip_streamwriter_with_errors.write()
        self._roundtrip_stringstream_with_errors.seek(0)  # return to beginning
        _read_parametric_model = self._roundtrip_streamreader_with_errors.read()
        self.assertTrue(isinstance(_read_parametric_model, XYParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        _read_parametric_model.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _read_parametric_model.y,
                self._test_parametric_model_with_errors.y
            )
        )
        
        _given_error = self._test_parametric_model_with_errors.get_error('test_x_error')
        _read_error = _read_parametric_model.get_error('test_x_error')
        self.assertTrue(_given_error['axis'] == _read_error['axis'])
        self.assertTrue(_given_error['enabled'] == _read_error['enabled'])
        self.assertTrue(
            np.allclose(
                _given_error['err'].error,
                _read_error['err'].error
            )
        )
        self.assertTrue(_given_error['err'].corr_coeff == _read_error['err'].corr_coeff)