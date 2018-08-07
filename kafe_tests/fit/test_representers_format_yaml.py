import unittest
from six import StringIO

from kafe.fit._base import ModelParameterFormatter
from kafe.fit.histogram import HistModelDensityFunctionFormatter
from kafe.fit.indexed import IndexedModelFunctionFormatter
from kafe.fit.xy import XYModelFunctionFormatter
from kafe.fit.xy_multi import XYMultiModelFunctionFormatter
from kafe.fit.representation import ModelFunctionFormatterYamlWriter, ModelFunctionFormatterYamlReader
from kafe.fit.representation import ModelParameterFormatterYamlWriter, ModelParameterFormatterYamlReader
from kafe.fit.io.handle import IOStreamHandle


TEST_MODEL_FUNCTION_FORMATTER_HISTOGRAM="""
model_function_formatter:
    type: histogram
    name: quadratic_model
    latex_name: quadratic model
    x_name: x
    latex_x_name: X
    arg_formatters:
        - model_parameter_formatter:
            name: a
            latex_name: A
        - model_parameter_formatter:
            name: b
            latex_name: B
        - model_parameter_formatter:
            name: c
            latex_name: C
    expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
    latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
"""

class TestHistModelFunctionFormatterYamlRepresenter(unittest.TestCase):

    def setUp(self):
        self._model_function_formatter = HistModelDensityFunctionFormatter(
            name='quadratic_model',
            latex_name='quadratic model',
            x_name='x',
            latex_x_name='X',
            arg_formatters=[
                ModelParameterFormatter(name='a', value=1.1, error=0.1, latex_name='A'),
                ModelParameterFormatter(name='b', value=2.2, error=0.1, latex_name='B'),
                ModelParameterFormatter(name='c', value=3.3, error=0.1, latex_name='C')
            ],
            expression_string='{0} * {x} ** 2 + {1} * {x} + {2}',
            latex_expression_string='{0}{x}^2 + {1}{x} + {2}' 
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_HISTOGRAM))

        self._roundtrip_streamreader = ModelFunctionFormatterYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionFormatterYamlWriter(self._model_function_formatter, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionFormatterYamlReader(self._testfile_stringstream)
    
    def _assert_model_function_formatters_equal(self, formatter1, formatter2):
        self.assertTrue(formatter1.name == formatter2.name)
        self.assertTrue(formatter1.latex_name == formatter2.latex_name)
        self.assertTrue(formatter1._x_name == formatter2._x_name)
        self.assertTrue(formatter1._latex_x_name == formatter2._latex_x_name)
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
        self.assertTrue(isinstance(_read_model_function_formatter, HistModelDensityFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function_formatter = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, HistModelDensityFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

TEST_MODEL_FUNCTION_FORMATTER_INDEXED="""
model_function_formatter:
    type: indexed
    name: quadratic_model
    latex_name: quadratic model
    index_name: x
    latex_index_name: X
    arg_formatters:
        - model_parameter_formatter:
            name: a
            latex_name: A
        - model_parameter_formatter:
            name: b
            latex_name: B
        - model_parameter_formatter:
            name: c
            latex_name: C
    expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
    latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
"""

class TestIndexedModelFunctionFormatterYamlRepresenter(unittest.TestCase):

    def setUp(self):
        self._model_function_formatter = IndexedModelFunctionFormatter(
            name='quadratic_model',
            latex_name='quadratic model',
            index_name='x',
            latex_index_name='X',
            arg_formatters=[
                ModelParameterFormatter(name='a', value=1.1, error=0.1, latex_name='A'),
                ModelParameterFormatter(name='b', value=2.2, error=0.1, latex_name='B'),
                ModelParameterFormatter(name='c', value=3.3, error=0.1, latex_name='C')
            ],
            expression_string='{0} * {x} ** 2 + {1} * {x} + {2}',
            latex_expression_string='{0}{x}^2 + {1}{x} + {2}' 
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_INDEXED))

        self._roundtrip_streamreader = ModelFunctionFormatterYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionFormatterYamlWriter(self._model_function_formatter, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionFormatterYamlReader(self._testfile_stringstream)
    
    def _assert_model_function_formatters_equal(self, formatter1, formatter2):
        self.assertTrue(formatter1.name == formatter2.name)
        self.assertTrue(formatter1.latex_name == formatter2.latex_name)
        self.assertTrue(formatter1._index_name == formatter2._index_name)
        self.assertTrue(formatter1._latex_index_name == formatter2._latex_index_name)
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

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function_formatter = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, IndexedModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

TEST_MODEL_FUNCTION_FORMATTER_XY="""
model_function_formatter:
    type: xy
    name: quadratic_model
    latex_name: quadratic model
    x_name: x
    latex_x_name: X
    arg_formatters:
        - model_parameter_formatter:
            name: a
            latex_name: A
        - model_parameter_formatter:
            name: b
            latex_name: B
        - model_parameter_formatter:
            name: c
            latex_name: C
    expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
    latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
"""

class TestXYModelFunctionFormatterYamlRepresenter(unittest.TestCase):

    def _assert_model_function_formatters_equal(self, formatter1, formatter2):
        self.assertTrue(formatter1.name == formatter2.name)
        self.assertTrue(formatter1.latex_name == formatter2.latex_name)
        self.assertTrue(formatter1._x_name == formatter2._x_name)
        self.assertTrue(formatter1._latex_x_name == formatter2._latex_x_name)
        for _arg_formatter_1, _arg_formatter_2 in zip(
                formatter1._arg_formatters, formatter2._arg_formatters):
            self.assertTrue(_arg_formatter_1.name == _arg_formatter_2.name)
            self.assertTrue(_arg_formatter_1.latex_name == _arg_formatter_2.latex_name)
        self.assertTrue(formatter1.expression_format_string == formatter2.expression_format_string)
        self.assertTrue(formatter1.latex_expression_format_string == formatter2.latex_expression_format_string)

    def setUp(self):
        self._model_function_formatter = XYModelFunctionFormatter(
            name='quadratic_model',
            latex_name='quadratic model',
            x_name='x',
            latex_x_name='X',
            arg_formatters=[
                ModelParameterFormatter(name='a', value=1.1, error=0.1, latex_name='A'),
                ModelParameterFormatter(name='b', value=2.2, error=0.1, latex_name='B'),
                ModelParameterFormatter(name='c', value=3.3, error=0.1, latex_name='C')
            ],
            expression_string='{0} * {x} ** 2 + {1} * {x} + {2}',
            latex_expression_string='{0}{x}^2 + {1}{x} + {2}' 
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_XY))

        self._roundtrip_streamreader = ModelFunctionFormatterYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionFormatterYamlWriter(self._model_function_formatter, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionFormatterYamlReader(self._testfile_stringstream)
    
    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function_formatter = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, XYModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function_formatter = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, XYModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

TEST_MODEL_FUNCTION_FORMATTER_XY_MULTI="""
model_function_formatter:
    type: xy_multi
    singular_formatters:
        model_function_formatter:
            type: xy
            name: linear_model
            latex_name: linear model
            x_name: x
            latex_x_name: X
            expression_string: '{0} * {x} + {1}'
            latex_expression_string: '{0}{x} + {1}' 
        model_function_formatter:
            type: xy
            name: quadratic_model
            latex_name: quadratic model
            x_name: x
            latex_x_name: X
            expression_string: '{0} * {x} ** 2 + {1} * {x} + {2}'
            latex_expression_string: '{0}{x}^2 + {1}{x} + {2}' 
    arg_formatters:
        - model_parameter_formatter:
            name: a
            latex_name: A
        - model_parameter_formatter:
            name: b
            latex_name: B
        - model_parameter_formatter:
            name: c
            latex_name: C
"""

class TestXYMultiModelFunctionFormatterYamlRepresenter(unittest.TestCase):

    def setUp(self):
        _arg_formatters=[
            ModelParameterFormatter(name='a', value=1.1, error=0.1, latex_name='A'),
            ModelParameterFormatter(name='b', value=2.2, error=0.1, latex_name='B'),
            ModelParameterFormatter(name='c', value=3.3, error=0.1, latex_name='C')
        ]
        self._singular_formatter_1 = XYModelFunctionFormatter(
            name='linear_model',
            latex_name='linear model',
            x_name='x',
            latex_x_name='X',
            arg_formatters=_arg_formatters,
            expression_string='{0} * {x} + {1}',
            latex_expression_string='{0}{x}+ {1}' 
        )
        self._singular_formatter_2 = XYModelFunctionFormatter(
            name='quadratic_model',
            latex_name='quadratic model',
            x_name='x',
            latex_x_name='X',
            arg_formatters=_arg_formatters,
            expression_string='{0} * {x} ** 2 + {1} * {x} + {2}',
            latex_expression_string='{0}{x}^2 + {1}{x} + {2}' 
        )
        self._model_function_formatter = XYMultiModelFunctionFormatter(
            singular_formatters=[self._singular_formatter_1, self._singular_formatter_2],
            arg_formatters=_arg_formatters
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_FUNCTION_FORMATTER_XY_MULTI))

        self._roundtrip_streamreader = ModelFunctionFormatterYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelFunctionFormatterYamlWriter(self._model_function_formatter, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelFunctionFormatterYamlReader(self._testfile_stringstream)
    
    def _assert_model_function_formatters_equal(self, formatter_1, formatter_2):
        for _arg_formatter_1, _arg_formatter_2 in zip(
                formatter_1._arg_formatters, formatter_2._arg_formatters):
            self.assertTrue(_arg_formatter_1.name == _arg_formatter_2.name)
            self.assertTrue(_arg_formatter_1.latex_name == _arg_formatter_2.latex_name)
        for _singular_formatter_1, _singular_formatter_2 in zip(
                formatter_1._singular_formatters, formatter_2._singular_formatters):
            self.assertTrue(_singular_formatter_1.name == _singular_formatter_2.name)
            self.assertTrue(_singular_formatter_1.latex_name == _singular_formatter_2.latex_name)
            self.assertTrue(_singular_formatter_1._x_name == _singular_formatter_2._x_name)
            self.assertTrue(_singular_formatter_1._latex_x_name == _singular_formatter_2._latex_x_name)
            self.assertTrue(_singular_formatter_1.expression_format_string == _singular_formatter_2.expression_format_string)
            self.assertTrue(_singular_formatter_1.latex_expression_format_string == _singular_formatter_2.latex_expression_format_string)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_function_formatter = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, XYMultiModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function_formatter = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function_formatter, XYMultiModelFunctionFormatter))
        self._assert_model_function_formatters_equal(_read_model_function_formatter, self._model_function_formatter)

TEST_MODEL_PARAMETER_FORMATTER = """
model_parameter_formatter:
    name: phi
    latex_name: \phi
"""

class TestModelParameterFormatterYamlRepresenter(unittest.TestCase):

    def setUp(self):
        self._model_parameter_formatter = ModelParameterFormatter(
            name='phi',
            value=1.571,
            error=0.1,
            latex_name=r"\phi"
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MODEL_PARAMETER_FORMATTER))

        self._roundtrip_streamreader = ModelParameterFormatterYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ModelParameterFormatterYamlWriter(self._model_parameter_formatter, self._roundtrip_stringstream)
        self._testfile_streamreader = ModelParameterFormatterYamlReader(self._testfile_stringstream)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_model_parameter_formatter = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_parameter_formatter, ModelParameterFormatter))
        
        self.assertTrue(_read_model_parameter_formatter.name == self._model_parameter_formatter.name)
        self.assertTrue(_read_model_parameter_formatter.value == None)
        self.assertTrue(_read_model_parameter_formatter.error == None)
        self.assertTrue(_read_model_parameter_formatter.latex_name == self._model_parameter_formatter.latex_name)

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_parameter_formatter = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_parameter_formatter, ModelParameterFormatter))

        self.assertTrue(_read_parameter_formatter.name == self._model_parameter_formatter.name)
        self.assertTrue(_read_parameter_formatter.value == None)
        self.assertTrue(_read_parameter_formatter.error == None)
        self.assertTrue(_read_parameter_formatter.latex_name == self._model_parameter_formatter.latex_name)
