import unittest2 as unittest
import numpy as np
from six import StringIO

from kafe2.fit.histogram import HistModelFunction, HistModelDensityFunctionFormatter
from kafe2.fit.indexed import IndexedModelFunction, IndexedModelFunctionFormatter
from kafe2.fit.xy import XYModelFunction, XYModelFunctionFormatter
from kafe2.fit.xy_multi import XYMultiModelFunction, XYMultiModelFunctionFormatter
from kafe2.fit.representation import ModelFunctionYamlWriter, ModelFunctionYamlReader
from kafe2.fit.io.handle import IOStreamHandle
from kafe2.fit.representation._yaml_base import YamlReaderException

TEST_MODEL_FUNCTION_HIST="""
type: histogram
python_code: |
    def linear_model(x, a, b):
        return a * x + b
"""

TEST_MODEL_FUNCTION_HIST_MISSING_KEYWORD="""
type: histogram
"""

TEST_MODEL_FUNCTION_HIST_EXTRA_KEYWORD = TEST_MODEL_FUNCTION_HIST + """
extra_keyword: 3.14
"""

TEST_MODEL_FUNCTION_HIST_WITH_FORMATTER = TEST_MODEL_FUNCTION_HIST + r"""
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

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_HIST_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_HIST_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ModelFunctionYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ModelFunctionYamlReader(self._testfile_stringstream_extra_keyword)

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

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

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
        self.assertTrue(_read_arg_formatters[0].latex_name == r'{\alpha}')
        self.assertTrue(_read_arg_formatters[1].name == 'beta')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'{\beta}')
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

TEST_MODEL_FUNCTION_INDEXED = """
type: indexed
python_code: |
    def linear_model(a, b):
        return a * np.arange(10) + b
"""

TEST_MODEL_FUNCTION_INDEXED_MISSING_KEYWORD = """
type: indexed
"""

TEST_MODEL_FUNCTION_INDEXED_EXTRA_KEYWORD = TEST_MODEL_FUNCTION_INDEXED + """
extra_keyword: 3.14
"""

TEST_MODEL_FUNCTION_INDEXED_WITH_FORMATTER = TEST_MODEL_FUNCTION_INDEXED + r"""
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

    TEST_X = np.arange(10)

    @staticmethod
    def linear_model(a, b):
        #TODO handle
        #return a * TestIndexedModelFunctionYamlRepresenter.TEST_X + b
        return a * np.arange(10) + b
    
    def setUp(self):
        self._test_a = 2.0
        self._test_b = -1.0
        self._test_y = self.linear_model(self._test_a, self._test_b)
        
        self._model_function = IndexedModelFunction(self.linear_model)
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_INDEXED))
        self._testfile_stringstream_with_formatter = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_INDEXED_WITH_FORMATTER))
        
        self._roundtrip_streamreader = ModelFunctionYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionYamlWriter(self._model_function, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_formatter = ModelFunctionYamlReader(self._testfile_stringstream_with_formatter)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_INDEXED_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_INDEXED_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ModelFunctionYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ModelFunctionYamlReader(self._testfile_stringstream_extra_keyword)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, IndexedModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_a, self._test_b),
                self._test_y
            )
        )

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_read_from_testfile_stream_with_formatter(self):
        _read_model_function = self._testfile_streamreader_with_formatter.read()
        self.assertTrue(isinstance(_read_model_function, IndexedModelFunction))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_a, self._test_b),
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
        self.assertTrue(_read_arg_formatters[0].latex_name == r'{\alpha}')
        self.assertTrue(_read_arg_formatters[1].name == 'beta')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'{\beta}')
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
                _read_model_function.func(self._test_a, self._test_b)
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

TEST_MODEL_FUNCTION_XY_MISSING_KEYWORD="""
type: xy
"""

TEST_MODEL_FUNCTION_XY_EXTRA_KEYWORD = TEST_MODEL_FUNCTION_XY + """
extra_keyword: 3.14
"""

TEST_MODEL_FUNCTION_XY_WITH_FORMATTER = TEST_MODEL_FUNCTION_XY + r"""
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

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ModelFunctionYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ModelFunctionYamlReader(self._testfile_stringstream_extra_keyword)

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

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

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
        self.assertTrue(_read_arg_formatters[0].latex_name == r'{\alpha}')
        self.assertTrue(_read_arg_formatters[1].name == 'beta')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'{\beta}')
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

TEST_MODEL_FUNCTION_XY_MULTI_MISSING_KEYWORD="""
type: xy_multi
python_code:
  - |
    def quadratic_model(x, a, b, c):
        return a * x ** 2 + b * x + c
  - |
    def linear_model(x, b, c):
        return b * x + c
"""

TEST_MODEL_FUNCTION_XY_MULTI_EXTRA_KEYWORD = TEST_MODEL_FUNCTION_XY_MULTI + """
extra_keyword: 3.14
"""

TEST_MODEL_FUNCTION_XY_MULTI_WITH_FORMATTER = TEST_MODEL_FUNCTION_XY_MULTI + r"""
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

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_MULTI_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_MULTI_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ModelFunctionYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ModelFunctionYamlReader(self._testfile_stringstream_extra_keyword)

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

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

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
        self.assertTrue(_read_arg_formatters[0].latex_name == r'{\alpha}')
        self.assertTrue(_read_arg_formatters[1].name == 'beta')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'{\beta}')
        self.assertTrue(_read_arg_formatters[2].name == 'gamma')
        self.assertTrue(_read_arg_formatters[2].latex_name == r'{\gamma}')

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