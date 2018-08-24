import unittest
import numpy as np
from six import StringIO

from kafe.fit.histogram import HistModelFunction, HistModelDensityFunctionFormatter
from kafe.fit.indexed import IndexedModelFunction, IndexedModelFunctionFormatter
from kafe.fit.xy import XYModelFunction, XYModelFunctionFormatter
from kafe.fit.xy_multi import XYMultiModelFunction, XYMultiModelFunctionFormatter
from kafe.fit.representation import ModelFunctionYamlWriter, ModelFunctionYamlReader
from kafe.fit.representation import ParametricModelYamlWriter, ParametricModelYamlReader
from kafe.fit.io.handle import IOStreamHandle
from kafe.fit.xy.model import XYParametricModel

TEST_MODEL_FUNCTION_HIST="""
type: histogram
python_code: |
    def linear_model(x, a, b):
        return a * x + b
"""

TEST_MODEL_FUNCTION_HIST_WITH_FORMATTER=r"""
type: histogram
python_code: |
    def linear_model(x, a, b):
        return a * x + b
model_function_formatter:
    type: histogram
    name: linear_model
    latex_name: linear model
    x_name: r
    latex_x_name: r
    arg_formatters:
      - name: alpha
        latex_name: \alpha
      - name: beta
        latex_name: \beta
    expression_string: '{0} * {x} + {1}'
    latex_expression_string: '{0}{x} + {1}' 
"""

class TestHistModelFunctionYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(x, a, b):
        return a * x + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_a = 2.0
        self._test_b = -1.0
        self._test_y = self.linear_model(self._test_x, self._test_a, self._test_b)
        
        self._model_function = HistModelFunction(self.linear_model)
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_HIST))
        self._testfile_stringstream_with_formatter = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_HIST_WITH_FORMATTER))
        
        self._roundtrip_streamreader = ModelFunctionYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionYamlWriter(self._model_function, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_formatter = ModelFunctionYamlReader(self._testfile_stringstream_with_formatter)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, HistModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_x, self._test_a, self._test_b),
                self._test_y
            )
        )

    def test_read_from_testfile_stream_with_formatter(self):
        _read_model_function = self._testfile_streamreader_with_formatter.read()
        self.assertTrue(isinstance(_read_model_function, HistModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_x, self._test_a, self._test_b),
                self._test_y
            )
        )
        _read_formatter = _read_model_function.formatter
        self.assertTrue(isinstance(_read_formatter, HistModelDensityFunctionFormatter))
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
        self.assertTrue(isinstance(_read_model_function, HistModelFunction))

        self.assertTrue(
            np.allclose(
                self._test_y,
                _read_model_function.func(self._test_x, self._test_a, self._test_b)
            )
        )
        
        _given_formatter = self._model_function.formatter
        _read_formatter = _read_model_function.formatter
        
        self.assertTrue(isinstance(_read_formatter, HistModelDensityFunctionFormatter))
        
        _given_arg_formatters = _given_formatter.arg_formatters
        _read_arg_formatters = _read_formatter.arg_formatters
        
        self.assertTrue(_read_formatter.name == _given_formatter.name)
        self.assertTrue(_read_formatter.latex_name == _given_formatter.latex_name)
        self.assertTrue(_read_formatter._x_name == _given_formatter._x_name)
        self.assertTrue(_read_formatter._latex_x_name == _given_formatter._latex_x_name)
        self.assertTrue(_read_arg_formatters[0].name == _given_arg_formatters[0].name)
        self.assertTrue(_read_arg_formatters[0].latex_name == _given_arg_formatters[0].latex_name)
        self.assertTrue(_read_arg_formatters[1].name == _given_arg_formatters[1].name)
        self.assertTrue(_read_arg_formatters[1].latex_name == _given_arg_formatters[1].latex_name)
        self.assertTrue(_read_formatter.expression_format_string ==  _given_formatter.expression_format_string)
        self.assertTrue(_read_formatter.latex_expression_format_string ==  _given_formatter.latex_expression_format_string)

TEST_MODEL_FUNCTION_INDEXED="""
type: indexed
python_code: |
    def linear_model(x, a, b):
        return a * x + b
"""

TEST_MODEL_FUNCTION_INDEXED_WITH_FORMATTER=r"""
type: indexed
python_code: |
    def linear_model(x, a, b):
        return a * x + b
model_function_formatter:
    type: indexed
    name: linear_model
    latex_name: linear model
    index_name: r
    latex_index_name: r
    arg_formatters:
      - name: alpha
        latex_name: \alpha
      - name: beta
        latex_name: \beta
    expression_string: '{0} * {x} + {1}'
    latex_expression_string: '{0}{x} + {1}' 
"""

class TestIndexedModelFunctionYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(x, a, b):
        return a * x + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_a = 2.0
        self._test_b = -1.0
        self._test_y = self.linear_model(self._test_x, self._test_a, self._test_b)
        
        self._model_function = IndexedModelFunction(self.linear_model)
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_INDEXED))
        self._testfile_stringstream_with_formatter = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_INDEXED_WITH_FORMATTER))
        
        self._roundtrip_streamreader = ModelFunctionYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionYamlWriter(self._model_function, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_formatter = ModelFunctionYamlReader(self._testfile_stringstream_with_formatter)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, IndexedModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_x, self._test_a, self._test_b),
                self._test_y
            )
        )

    def test_read_from_testfile_stream_with_formatter(self):
        _read_model_function = self._testfile_streamreader_with_formatter.read()
        self.assertTrue(isinstance(_read_model_function, IndexedModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_x, self._test_a, self._test_b),
                self._test_y
            )
        )
        _read_formatter = _read_model_function.formatter
        self.assertTrue(isinstance(_read_formatter, IndexedModelFunctionFormatter))
        _read_arg_formatters = _read_formatter.arg_formatters
        self.assertTrue(_read_formatter.name == 'linear_model')
        self.assertTrue(_read_formatter.latex_name == 'linear model')
        self.assertTrue(_read_formatter._index_name == 'r')
        self.assertTrue(_read_formatter._latex_index_name == 'r')
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
        self.assertTrue(isinstance(_read_model_function, IndexedModelFunction))

        self.assertTrue(
            np.allclose(
                self._test_y,
                _read_model_function.func(self._test_x, self._test_a, self._test_b)
            )
        )
        
        _given_formatter = self._model_function.formatter
        _read_formatter = _read_model_function.formatter
        
        self.assertTrue(isinstance(_read_formatter, IndexedModelFunctionFormatter))
        
        _given_arg_formatters = _given_formatter.arg_formatters
        _read_arg_formatters = _read_formatter.arg_formatters
        
        self.assertTrue(_read_formatter.name == _given_formatter.name)
        self.assertTrue(_read_formatter.latex_name == _given_formatter.latex_name)
        self.assertTrue(_read_formatter._index_name == _given_formatter._index_name)
        self.assertTrue(_read_formatter._latex_index_name == _given_formatter._latex_index_name)
        self.assertTrue(_read_arg_formatters[0].name == _given_arg_formatters[0].name)
        self.assertTrue(_read_arg_formatters[0].latex_name == _given_arg_formatters[0].latex_name)
        self.assertTrue(_read_arg_formatters[1].name == _given_arg_formatters[1].name)
        self.assertTrue(_read_arg_formatters[1].latex_name == _given_arg_formatters[1].latex_name)
        self.assertTrue(_read_formatter.expression_format_string ==  _given_formatter.expression_format_string)
        self.assertTrue(_read_formatter.latex_expression_format_string ==  _given_formatter.latex_expression_format_string)

TEST_MODEL_FUNCTION_XY="""
type: xy
python_code: |
    def linear_model(x, a, b):
        return a * x + b
"""

TEST_MODEL_FUNCTION_XY_WITH_FORMATTER=r"""
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
      - name: alpha
        latex_name: \alpha
      - name: beta
        latex_name: \beta
    expression_string: '{0} * {x} + {1}'
    latex_expression_string: '{0}{x} + {1}' 
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

        self.assertTrue(
            np.allclose(
                self._test_y,
                _read_model_function.func(self._test_x, self._test_a, self._test_b)
            )
        )
        
        _given_formatter = self._model_function.formatter
        _read_formatter = _read_model_function.formatter
        
        self.assertTrue(isinstance(_read_formatter, XYModelFunctionFormatter))
        
        _given_arg_formatters = _given_formatter.arg_formatters
        _read_arg_formatters = _read_formatter.arg_formatters
        
        self.assertTrue(_read_formatter.name == _given_formatter.name)
        self.assertTrue(_read_formatter.latex_name == _given_formatter.latex_name)
        self.assertTrue(_read_formatter._x_name == _given_formatter._x_name)
        self.assertTrue(_read_formatter._latex_x_name == _given_formatter._latex_x_name)
        self.assertTrue(_read_arg_formatters[0].name == _given_arg_formatters[0].name)
        self.assertTrue(_read_arg_formatters[0].latex_name == _given_arg_formatters[0].latex_name)
        self.assertTrue(_read_arg_formatters[1].name == _given_arg_formatters[1].name)
        self.assertTrue(_read_arg_formatters[1].latex_name == _given_arg_formatters[1].latex_name)
        self.assertTrue(_read_formatter.expression_format_string ==  _given_formatter.expression_format_string)
        self.assertTrue(_read_formatter.latex_expression_format_string ==  _given_formatter.latex_expression_format_string)

TEST_MODEL_FUNCTION_XY_MULTI="""
type: xy_multi
python_code:
  - |
    def quadratic_model(x, a, b, c):
        return a * x ** 2 + b * x + c
  - |
    def linear_model(x, b, c):
        return b * x + c
data_indices: [0, 10, 20]
"""

TEST_MODEL_FUNCTION_XY_MULTI_WITH_FORMATTER=r"""
type: xy_multi
python_code:
  - |
    def quadratic_model(x, a, b, c):
        return a * x ** 2 + b * x + c
  - |
    def linear_model(x, b, c):
        return b * x + c
data_indices: [0, 10, 20]
model_function_formatter:
    type: xy_multi
    singular_formatters:
      - type: xy
        name: quadratic_model
        latex_name: quadratic model
        x_name: r
        latex_x_name: r
        expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
        latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
      - type: xy
        name: linear_model
        latex_name: linear model
        x_name: r
        latex_x_name: r
        expression_string: '{0} * {x} + {1}'
        latex_expression_string: '{0}{x} + {1}' 
    arg_formatters:
      - name: alpha
        latex_name: \alpha
      - name: beta
        latex_name: \beta
      - name: gamma
        latex_name: \gamma
"""

class TestXYMultiModelFunctionYamlRepresenter(unittest.TestCase):

    @staticmethod
    def quadratic_model(x, a, b, c):
        return a * x ** 2 + b * x + c
    
    @staticmethod
    def linear_model(x, b, c):
        return b * x + c
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_x_combined = np.concatenate([self._test_x, self._test_x])
        self._test_a = 2.0
        self._test_b = -1.0
        self._test_c = 0.5
        self._test_y = np.concatenate([
            self.quadratic_model(self._test_x, self._test_a, self._test_b, self._test_c),
            self.linear_model(self._test_x, self._test_b, self._test_c)
        ])
        
        self._model_function = XYMultiModelFunction(
            model_function_list=(self.quadratic_model, self.linear_model), 
            data_indices=[0, 10, 20]
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_MULTI))
        self._testfile_stringstream_with_formatter = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_MULTI_WITH_FORMATTER))
        
        self._roundtrip_streamreader = ModelFunctionYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionYamlWriter(self._model_function, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_formatter = ModelFunctionYamlReader(self._testfile_stringstream_with_formatter)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, XYMultiModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function._eval(self._test_x_combined, self._test_a, self._test_b, self._test_c),
                self._test_y
            )
        )

    def test_read_from_testfile_stream_with_formatter(self):
        _read_model_function = self._testfile_streamreader_with_formatter.read()
        self.assertTrue(isinstance(_read_model_function, XYMultiModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function._eval(self._test_x_combined, self._test_a, self._test_b, self._test_c),
                self._test_y
            )
        )
        _read_formatter = _read_model_function.formatter
        self.assertTrue(isinstance(_read_formatter, XYMultiModelFunctionFormatter))
        _read_arg_formatters = _read_formatter.arg_formatters
        _singular_formatter_0 = _read_formatter._singular_formatters[0]
        _singular_formatter_1 = _read_formatter._singular_formatters[1]
        self.assertTrue(_singular_formatter_0.name == 'quadratic_model')
        self.assertTrue(_singular_formatter_0.latex_name == 'quadratic model')
        self.assertTrue(_singular_formatter_0._x_name == 'r')
        self.assertTrue(_singular_formatter_0._latex_x_name == 'r')
        self.assertTrue(_singular_formatter_0.expression_format_string == '{0} * {x} ** 2 + {1} * {x} + {2}')
        self.assertTrue(_singular_formatter_0.latex_expression_format_string == '{0}{x}^2 + {1}{x} + {2}')
        self.assertTrue(_singular_formatter_1.name == 'linear_model')
        self.assertTrue(_singular_formatter_1.latex_name == 'linear model')
        self.assertTrue(_singular_formatter_1._x_name == 'r')
        self.assertTrue(_singular_formatter_1._latex_x_name == 'r')
        self.assertTrue(_singular_formatter_1.expression_format_string == '{0} * {x} + {1}')
        self.assertTrue(_singular_formatter_1.latex_expression_format_string == '{0}{x} + {1}')
        self.assertTrue(_read_arg_formatters[0].name == 'alpha')
        self.assertTrue(_read_arg_formatters[0].latex_name == r'\alpha')
        self.assertTrue(_read_arg_formatters[1].name == 'beta')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'\beta')
        self.assertTrue(_read_arg_formatters[2].name == 'gamma')
        self.assertTrue(_read_arg_formatters[2].latex_name == r'\gamma')

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, XYMultiModelFunction))

        self.assertTrue(
            np.allclose(
                _read_model_function._eval(self._test_x_combined, self._test_a, self._test_b, self._test_c),
                self._test_y
            )
        )
        self.assertTrue(isinstance(_read_model_function.formatter, XYMultiModelFunctionFormatter))
        
        _given_formatter_0 = self._model_function.formatter._singular_formatters[0]
        _read_formatter_0 = _read_model_function.formatter._singular_formatters[0]
        _given_formatter_1 = self._model_function.formatter._singular_formatters[1]
        _read_formatter_1 = _read_model_function.formatter._singular_formatters[1]
        _given_arg_formatters = self._model_function.formatter.arg_formatters
        _read_arg_formatters = _read_model_function.formatter.arg_formatters
        
        self.assertTrue(_read_formatter_0.name == _given_formatter_0.name)
        self.assertTrue(_read_formatter_0.latex_name == _given_formatter_0.latex_name)
        self.assertTrue(_read_formatter_0._x_name == _given_formatter_0._x_name)
        self.assertTrue(_read_formatter_0._latex_x_name == _given_formatter_0._latex_x_name)
        self.assertTrue(_read_formatter_0.expression_format_string ==  _given_formatter_0.expression_format_string)
        self.assertTrue(_read_formatter_0.latex_expression_format_string ==  _given_formatter_0.latex_expression_format_string)
        self.assertTrue(_read_formatter_1.name == _given_formatter_1.name)
        self.assertTrue(_read_formatter_1.latex_name == _given_formatter_1.latex_name)
        self.assertTrue(_read_formatter_1._x_name == _given_formatter_1._x_name)
        self.assertTrue(_read_formatter_1._latex_x_name == _given_formatter_1._latex_x_name)
        self.assertTrue(_read_formatter_1.expression_format_string ==  _given_formatter_1.expression_format_string)
        self.assertTrue(_read_formatter_1.latex_expression_format_string ==  _given_formatter_1.latex_expression_format_string)
        self.assertTrue(_read_arg_formatters[0].name == _given_arg_formatters[0].name)
        self.assertTrue(_read_arg_formatters[0].latex_name == _given_arg_formatters[0].latex_name)
        self.assertTrue(_read_arg_formatters[1].name == _given_arg_formatters[1].name)
        self.assertTrue(_read_arg_formatters[1].latex_name == _given_arg_formatters[1].latex_name)
        self.assertTrue(_read_arg_formatters[2].name == _given_arg_formatters[2].name)
        self.assertTrue(_read_arg_formatters[2].latex_name == _given_arg_formatters[2].latex_name)

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