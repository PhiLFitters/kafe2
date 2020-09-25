import unittest2 as unittest
from six import StringIO

from kafe2.fit._base import ParameterFormatter, ModelFunctionFormatter
from kafe2.fit.indexed import IndexedModelFunctionFormatter
from kafe2.fit.representation import ModelFunctionFormatterYamlWriter,\
    ModelFunctionFormatterYamlReader, ParameterFormatterYamlWriter, ParameterFormatterYamlReader
from kafe2.fit.io.handle import IOStreamHandle
from kafe2.fit.representation._yaml_base import YamlReaderException


TEST_MODEL_FUNCTION_FORMATTER = """
name: quadratic_model
latex_name: quadratic model
signature: [x, a, b, c]
arg_formatters:
  x: X
  a: A
  b: B
  c: C
expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
"""

TEST_MODEL_FUNCTION_FORMATTER_MISSING_KEYWORD = """
latex_name: quadratic model
signature: [x, a, b, c]
arg_formatters:
  x: X
  a: A
  b: B
  c: C
expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
"""

TEST_MODEL_FUNCTION_FORMATTER_EXTRA_KEYWORD = TEST_MODEL_FUNCTION_FORMATTER + """\nextra_keyword: 3.14\n"""


class TestModelFunctionFormatterYamlRepresenter(unittest.TestCase):

    def setUp(self):
        self._model_function_formatter = ModelFunctionFormatter(
            name='quadratic_model',
            latex_name='quadratic model',
            arg_formatters=[
                ParameterFormatter(arg_name='x', latex_name='X'),
                ParameterFormatter(arg_name='a', value=1.1, error=0.1, latex_name='A'),
                ParameterFormatter(arg_name='b', value=2.2, error=0.1, latex_name='B'),
                ParameterFormatter(arg_name='c', value=3.3, error=0.1, latex_name='C')
            ],
            expression_string='{0} * {x} ** 2 + {1} * {x} + {2}',
            latex_expression_string='{0}{x}^2 + {1}{x} + {2}' 
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER))

        self._roundtrip_streamreader = ModelFunctionFormatterYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionFormatterYamlWriter(self._model_function_formatter,
                                                                        self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionFormatterYamlReader(self._testfile_stringstream)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ModelFunctionFormatterYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ModelFunctionFormatterYamlReader(self._testfile_stringstream_extra_keyword)
    
    def _assert_model_function_formatters_equal(self, formatter1, formatter2):
        self.assertTrue(formatter1.name == formatter2.name)
        self.assertTrue(formatter1.latex_name == formatter2.latex_name)
        for _arg_formatter_1, _arg_formatter_2 in zip(
                formatter1._arg_formatters, formatter2._arg_formatters):
            self.assertTrue(_arg_formatter_1.name == _arg_formatter_2.name)
            self.assertTrue(_arg_formatter_1.latex_name == _arg_formatter_2.latex_name)
        self.assertTrue(formatter1.expression_format_string == formatter2.expression_format_string)
        self.assertTrue(formatter1.latex_expression_format_string == formatter2.latex_expression_format_string)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function_formatter = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, ModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function_formatter = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, ModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)


TEST_MODEL_FUNCTION_FORMATTER_INDEXED = """
type: indexed
name: quadratic_model
latex_name: quadratic model
index_name: x
latex_index_name: X
signature: [a, b, c]
arg_formatters:
  a: A
  b: B
  c: C
expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
"""

TEST_MODEL_FUNCTION_FORMATTER_INDEXED_MISSING_KEYWORD = """
type: indexed
latex_name: quadratic model
index_name: x
latex_index_name: X
signature: [a, b, c]
arg_formatters:
  a: A
  b: B
  c: C
expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
"""

TEST_MODEL_FUNCTION_FORMATTER_INDEXED_EXTRA_KEYWORD = TEST_MODEL_FUNCTION_FORMATTER_INDEXED + """
extra_keyword: 3.14
"""


class TestIndexedModelFunctionFormatterYamlRepresenter(unittest.TestCase):

    def setUp(self):
        self._model_function_formatter = IndexedModelFunctionFormatter(
            name='quadratic_model',
            latex_name='quadratic model',
            index_name='x',
            latex_index_name='X',
            arg_formatters=[
                ParameterFormatter(arg_name='a', value=1.1, error=0.1, latex_name='A'),
                ParameterFormatter(arg_name='b', value=2.2, error=0.1, latex_name='B'),
                ParameterFormatter(arg_name='c', value=3.3, error=0.1, latex_name='C')
            ],
            expression_string='{0} * {x} ** 2 + {1} * {x} + {2}',
            latex_expression_string='{0}{x}^2 + {1}{x} + {2}' 
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_INDEXED))

        self._roundtrip_streamreader = ModelFunctionFormatterYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionFormatterYamlWriter(self._model_function_formatter,
                                                                        self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionFormatterYamlReader(self._testfile_stringstream)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_INDEXED_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_INDEXED_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ModelFunctionFormatterYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ModelFunctionFormatterYamlReader(self._testfile_stringstream_extra_keyword)
    
    def _assert_model_function_formatters_equal(self, formatter1, formatter2):
        self.assertTrue(formatter1.name == formatter2.name)
        self.assertTrue(formatter1.latex_name == formatter2.latex_name)
        self.assertTrue(formatter1.index_name == formatter2.index_name)
        self.assertTrue(formatter1.latex_index_name == formatter2.latex_index_name)
        for _arg_formatter_1, _arg_formatter_2 in zip(
                formatter1._arg_formatters, formatter2._arg_formatters):
            self.assertTrue(_arg_formatter_1.name == _arg_formatter_2.name)
            self.assertTrue(_arg_formatter_1.latex_name == _arg_formatter_2.latex_name)
        self.assertTrue(formatter1.expression_format_string == formatter2.expression_format_string)
        self.assertTrue(formatter1.latex_expression_format_string == formatter2.latex_expression_format_string)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function_formatter = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, IndexedModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function_formatter = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, IndexedModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)


TEST_MODEL_PARAMETER_FORMATTER = """
id: phi
name: ph
latex_name: \phi
"""

TEST_MODEL_PARAMETER_FORMATTER_MISSING_KEYWORD = """
latex_name: \phi
"""

TEST_MODEL_PARAMETER_FORMATTER_EXTRA_KEYWORD = """
name: phi
latex_name: \phi
extra_keyword: 3.14
"""


class TestModelParameterFormatterYamlRepresenter(unittest.TestCase):

    def setUp(self):
        self._model_parameter_formatter = ParameterFormatter('phi',
            name='ph',
            value=1.571,
            error=0.1,
            latex_name=r"\phi"
        )

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_PARAMETER_FORMATTER))

        self._roundtrip_streamreader = ParameterFormatterYamlReader(
            self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ParameterFormatterYamlWriter(
            self._model_parameter_formatter,
            self._roundtrip_stringstream)
        self._testfile_streamreader = ParameterFormatterYamlReader(self._testfile_stringstream)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(
            StringIO(TEST_MODEL_PARAMETER_FORMATTER_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(
            StringIO(TEST_MODEL_PARAMETER_FORMATTER_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ParameterFormatterYamlReader(
            self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ParameterFormatterYamlReader(
            self._testfile_stringstream_extra_keyword)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_parameter_formatter = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_parameter_formatter, ParameterFormatter))

        self.assertTrue(
            _read_model_parameter_formatter.name == self._model_parameter_formatter.name)
        self.assertTrue(_read_model_parameter_formatter.value is None)
        self.assertTrue(_read_model_parameter_formatter.error is None)
        self.assertTrue(
            _read_model_parameter_formatter.latex_name == self._model_parameter_formatter.latex_name)

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_parameter_formatter = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_parameter_formatter, ParameterFormatter))

        self.assertTrue(_read_parameter_formatter.name == self._model_parameter_formatter.name)
        self.assertTrue(_read_parameter_formatter.value is None)
        self.assertTrue(_read_parameter_formatter.error is None)
        self.assertTrue(
            _read_parameter_formatter.latex_name == self._model_parameter_formatter.latex_name)
