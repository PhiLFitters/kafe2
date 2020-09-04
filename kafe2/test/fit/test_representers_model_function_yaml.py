import unittest2 as unittest
import numpy as np
from six import StringIO

from kafe2.fit._base import ModelFunctionFormatter, ModelFunctionBase
from kafe2.fit.histogram import HistModelFunction
from kafe2.fit.indexed import IndexedModelFunction, IndexedModelFunctionFormatter
from kafe2.fit.representation import ModelFunctionYamlWriter, ModelFunctionYamlReader
from kafe2.fit.io.handle import IOStreamHandle
from kafe2.fit.representation._yaml_base import YamlReaderException

TEST_MODEL_FUNCTION_HIST = """
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
    name: linear_model
    latex_name: linear model
    arg_formatters:
      x: r
      a:
        name: alpha
        latex_name: \alpha
      b:
        name: beta
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
        self.assertTrue(isinstance(_read_formatter, ModelFunctionFormatter))
        _read_arg_formatters = _read_formatter.arg_formatters
        self.assertTrue(_read_formatter.name == 'linear_model')
        self.assertTrue(_read_formatter.latex_name == 'linear model')
        self.assertTrue(_read_arg_formatters[0].name == 'x')
        self.assertTrue(_read_arg_formatters[0].latex_name == r'{r}')
        self.assertTrue(_read_arg_formatters[1].name == 'alpha')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'{\alpha}')
        self.assertTrue(_read_arg_formatters[2].name == 'beta')
        self.assertTrue(_read_arg_formatters[2].latex_name == r'{\beta}')
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
        
        self.assertTrue(isinstance(_read_formatter, ModelFunctionFormatter))
        
        _given_arg_formatters = _given_formatter.arg_formatters
        _read_arg_formatters = _read_formatter.arg_formatters
        
        self.assertTrue(_read_formatter.name == _given_formatter.name)
        self.assertTrue(_read_formatter.latex_name == _given_formatter.latex_name)
        self.assertTrue(_read_arg_formatters[0].name == _given_arg_formatters[0].name)
        self.assertTrue(_read_arg_formatters[0].latex_name == _given_arg_formatters[0].latex_name)
        self.assertTrue(_read_arg_formatters[1].name == _given_arg_formatters[1].name)
        self.assertTrue(_read_arg_formatters[1].latex_name == _given_arg_formatters[1].latex_name)
        self.assertTrue(_read_arg_formatters[2].name == _given_arg_formatters[2].name)
        self.assertTrue(_read_arg_formatters[2].latex_name == _given_arg_formatters[2].latex_name)
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
    name: linear_model
    latex_name: linear model
    index_name: r
    latex_index_name: r
    arg_formatters:
      a:
        name: alpha
        latex_name: \alpha
      b:
        name: beta
        latex_name: \beta
    expression_string: '{0} * {r} + {1}'
    latex_expression_string: '{0}{r} + {1}' 
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
        self.assertTrue(_read_formatter.index_name == 'r')
        self.assertTrue(_read_formatter.latex_index_name == r'{r}')
        self.assertTrue(_read_arg_formatters[0].name == 'alpha')
        self.assertTrue(_read_arg_formatters[0].latex_name == r'{\alpha}')
        self.assertTrue(_read_arg_formatters[1].name == 'beta')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'{\beta}')
        self.assertTrue(_read_formatter.expression_format_string == '{0} * {r} + {1}')
        self.assertTrue(_read_formatter.latex_expression_format_string == '{0}{r} + {1}')

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
        self.assertTrue(_read_formatter.index_name == _given_formatter.index_name)
        self.assertTrue(_read_formatter.latex_index_name == _given_formatter.latex_index_name)
        self.assertTrue(_read_arg_formatters[0].name == _given_arg_formatters[0].name)
        self.assertTrue(_read_arg_formatters[0].latex_name == _given_arg_formatters[0].latex_name)
        self.assertTrue(_read_arg_formatters[1].name == _given_arg_formatters[1].name)
        self.assertTrue(_read_arg_formatters[1].latex_name == _given_arg_formatters[1].latex_name)
        self.assertTrue(_read_formatter.expression_format_string ==  _given_formatter.expression_format_string)
        self.assertTrue(_read_formatter.latex_expression_format_string ==  _given_formatter.latex_expression_format_string)


TEST_MODEL_FUNCTION_BASE = """
python_code: |
    def linear_model(x, a, b):
        return a * x + b
"""

TEST_MODEL_FUNCTION_BASE_MISSING_KEYWORD = """
type: base
"""

TEST_MODEL_FUNCTION_XY_EXTRA_KEYWORD = TEST_MODEL_FUNCTION_BASE + """
extra_keyword: 3.14
"""

TEST_MODEL_FUNCTION_XY_WITH_FORMATTER = TEST_MODEL_FUNCTION_BASE + r"""
model_function_formatter:
    name: linear_model
    latex_name: linear model
    arg_formatters:
      x: r
      a:
        name: alpha
        latex_name: \alpha
      b:
        name: beta
        latex_name: \beta
    expression_string: '{0} * {x} + {1}'
    latex_expression_string: '{0}{x} + {1}' 
"""


class TestModelFunctionBaseYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(x, a, b):
        return a * x + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_a = 2.0
        self._test_b = -1.0
        self._test_y = self.linear_model(self._test_x, self._test_a, self._test_b)
        
        self._model_function = ModelFunctionBase(self.linear_model)
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_BASE))
        self._testfile_stringstream_with_formatter = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_WITH_FORMATTER))
        
        self._roundtrip_streamreader = ModelFunctionYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionYamlWriter(self._model_function, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_formatter = ModelFunctionYamlReader(self._testfile_stringstream_with_formatter)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_BASE_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_XY_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ModelFunctionYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ModelFunctionYamlReader(self._testfile_stringstream_extra_keyword)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, ModelFunctionBase))
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
        self.assertTrue(isinstance(_read_model_function, ModelFunctionBase))
        self.assertTrue(
            np.allclose(
                _read_model_function.func(self._test_x, self._test_a, self._test_b),
                self._test_y
            )
        )
        _read_formatter = _read_model_function.formatter
        self.assertTrue(isinstance(_read_formatter, ModelFunctionFormatter))
        _read_arg_formatters = _read_formatter.arg_formatters
        self.assertTrue(_read_formatter.name == 'linear_model')
        self.assertTrue(_read_formatter.latex_name == 'linear model')
        self.assertTrue(_read_arg_formatters[0].name == 'x')
        self.assertTrue(_read_arg_formatters[0].latex_name == r'{r}')
        self.assertTrue(_read_arg_formatters[1].name == 'alpha')
        self.assertTrue(_read_arg_formatters[1].latex_name == r'{\alpha}')
        self.assertTrue(_read_arg_formatters[2].name == 'beta')
        self.assertTrue(_read_arg_formatters[2].latex_name == r'{\beta}')
        self.assertTrue(_read_formatter.expression_format_string == '{0} * {x} + {1}')
        self.assertTrue(_read_formatter.latex_expression_format_string == '{0}{x} + {1}')

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, ModelFunctionBase))

        self.assertTrue(
            np.allclose(
                self._test_y,
                _read_model_function.func(self._test_x, self._test_a, self._test_b)
            )
        )
        
        _given_formatter = self._model_function.formatter
        _read_formatter = _read_model_function.formatter
        
        self.assertTrue(isinstance(_read_formatter, ModelFunctionFormatter))
        
        _given_arg_formatters = _given_formatter.arg_formatters
        _read_arg_formatters = _read_formatter.arg_formatters
        
        self.assertTrue(_read_formatter.name == _given_formatter.name)
        self.assertTrue(_read_formatter.latex_name == _given_formatter.latex_name)
        self.assertTrue(_read_arg_formatters[0].name == _given_arg_formatters[0].name)
        self.assertTrue(_read_arg_formatters[0].latex_name == _given_arg_formatters[0].latex_name)
        self.assertTrue(_read_arg_formatters[1].name == _given_arg_formatters[1].name)
        self.assertTrue(_read_arg_formatters[1].latex_name == _given_arg_formatters[1].latex_name)
        self.assertTrue(_read_formatter.expression_format_string ==  _given_formatter.expression_format_string)
        self.assertTrue(_read_formatter.latex_expression_format_string ==  _given_formatter.latex_expression_format_string)
