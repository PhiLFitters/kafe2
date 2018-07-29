import unittest
import numpy as np
from six import StringIO

from kafe.fit.xy import XYModelFunction, XYModelFunctionFormatter
from kafe.fit.representation import ModelFunctionYamlWriter, ModelFunctionYamlReader
from kafe.fit.representation import ParametricModelYamlWriter, ParametricModelYamlReader
from kafe.fit.io.handle import IOStreamHandle
from kafe.fit.xy.model import XYParametricModel

TEST_MODEL_FUNCTION_XY="""
model_function:
    type: xy
    python_code: |
        def linear_model(x, a, b):
            return a * x + b
        
"""

TEST_MODEL_FUNCTION_XY_WITH_FORMATTER=r"""
model_function:
    type: xy
    python_code: |
        def linear_model(x, a, b):
            return a * x + b
    model_function_formatter:
        type: xy
        name: linear_model
        latex_name: linear model
        x_name: r
        latex_x_name: r
        arg_formatters:
            - model_parameter_formatter:
                name: alpha
                latex_name: \alpha
            - model_parameter_formatter:
                name: beta
                latex_name: \beta
        expression_string: '{0} * {x} + {1}'
        latex_expression_string: '{0}{x} + {1}' 
        
"""

TEST_PARAMETRIC_MODEL_XY="""
parametric_model:
    type: xy
    x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    model_function:
        type: xy
        python_code: |
            def linear_model(x, a, b):
                return a * x + b
    model_parameters: [1.0, 1.0]
"""

class TestXYModelFunctionYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(x, a, b):
        return a * x + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_a = 2.0
        self._test_b = -1.0
        self._test_y = self.linear_model(self._test_x, self._test_a, self._test_b)
        
        self._model_function = XYModelFunction(self.linear_model)
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY))
        self._testfile_stringstream_with_formatter = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_WITH_FORMATTER))
        
        self._roundtrip_streamreader = ModelFunctionYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionYamlWriter(self._model_function, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_formatter = ModelFunctionYamlReader(self._testfile_stringstream_with_formatter)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, XYModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_x, self._test_a, self._test_b),
                self._test_y
            )
        )

    def test_read_from_testfile_stream_with_formatter(self):
        _read_model_function = self._testfile_streamreader_with_formatter.read()
        self.assertTrue(isinstance(_read_model_function, XYModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_x, self._test_a, self._test_b),
                self._test_y
            )
        )
        _read_formatter = _read_model_function.formatter
        self.assertTrue(isinstance(_read_formatter, XYModelFunctionFormatter))
        _read_arg_formatters = _read_formatter.arg_formatters
        self.assertTrue(_read_formatter.name == 'linear_model')
        self.assertTrue(_read_formatter.latex_name == 'linear model')
        self.assertTrue(_read_formatter._x_name == 'r')
        self.assertTrue(_read_formatter._latex_x_name == 'r')
        self.assertTrue(_read_arg_formatters[0].name == 'alpha')
        self.assertTrue(_read_arg_formatters[0].latex_name == r'\alpha')
        self.assertTrue(_read_arg_formatters[1].name == 'beta')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'\beta')
        self.assertTrue(_read_formatter.expression_format_string == '{0} * {x} + {1}')
        self.assertTrue(_read_formatter.latex_expression_format_string == '{0}{x} + {1}')

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, XYModelFunction))

        # compare y values
        self.assertTrue(
            np.allclose(
                self._test_y,
                _read_model_function.func(self._test_x, self._test_a, self._test_b)
            )
        )

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
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_XY))
        
        self._roundtrip_streamreader = ParametricModelYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ParametricModelYamlWriter(self._test_parametric_model, self._roundtrip_stringstream)
        self._testfile_streamreader = ParametricModelYamlReader(self._testfile_stringstream)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

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
        
    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_parametric_model = self._roundtrip_streamreader.read()
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
    