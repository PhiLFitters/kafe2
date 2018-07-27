import unittest
from six import StringIO

from kafe.fit._base import ModelParameterFormatter
from kafe.fit.xy import XYModelFunctionFormatter
from kafe.fit.representation import ModelFunctionFormatterYamlWriter, ModelFunctionFormatterYamlReader
from kafe.fit.representation import ModelParameterFormatterYamlWriter, ModelParameterFormatterYamlReader
from kafe.fit.io.handle import IOStreamHandle

TEST_MODEL_FUNCTION_FORMATTER_XY="""
model_function_formatter:
    type: xy
    name: quadratic_model
    latex_name: quadratic model
    x_name: x
    latex_x_name: x
    expression_string: {0} * {x} ** 2 + {1} * {x} + {2}
    latex_expression_string: {0}{x}^2 + {1}{x} + {2} 
"""

TEST_MODEL_PARAMETER_FORMATTER = """
model_parameter_formatter:
    name: phi
    value: 1.571
    error: 0.1
    latex_name: \phi
"""

class TestXYModelFunctionFormatterYamlRepresenter(unittest.TestCase):

    def setUp(self):
        self._model_function_formatter = XYModelFunctionFormatter(
                name='quadratic_model',
                latex_name='quadratic model',
                x_name='x',
                latex_x_name='x',
                arg_formatters=[
                    ModelParameterFormatter(name='a', value='1.1', error=None, latex_name='a'),
                    ModelParameterFormatter(name='b', value='2.2', error=None, latex_name='b'),
                    ModelParameterFormatter(name='c', value='3.3', error=None, latex_name='c')
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

class TestXYArgumentFormatterYamlRepresenter(unittest.TestCase):

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
        _read_model_function = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, ModelParameterFormatter))
        
        self.assertTrue(_read_model_function.name == self._model_parameter_formatter.name)
        self.assertTrue(_read_model_function.value == self._model_parameter_formatter.value)
        self.assertTrue(_read_model_function.error == self._model_parameter_formatter.error)
        self.assertTrue(_read_model_function.latex_name == self._model_parameter_formatter.latex_name)

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_model_function = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_model_function, ModelParameterFormatter))

        self.assertTrue(_read_model_function.name == self._model_parameter_formatter.name)
        self.assertTrue(_read_model_function.value == self._model_parameter_formatter.value)
        self.assertTrue(_read_model_function.error == self._model_parameter_formatter.error)
        self.assertTrue(_read_model_function.latex_name == self._model_parameter_formatter.latex_name)
