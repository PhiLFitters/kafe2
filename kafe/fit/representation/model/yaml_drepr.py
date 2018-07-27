import inspect
import StringIO
import textwrap
import tokenize
import yaml

from .._base import DReprError, DReprWriterMixin, DReprReaderMixin
from ._base import ModelFunctionDReprBase
from .. import _AVAILABLE_REPRESENTATIONS

class ModelFunctionYamlWriter(DReprWriterMixin, ModelFunctionDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'writer'

    def __init__(self, model_function, output_io_handle):
        super(ModelFunctionYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_function=model_function)
    
    @staticmethod
    def _make_representation(model_function):
        _yaml = dict()

        # -- determine model function type
        _type = ModelFunctionYamlWriter._MODEL_FUNCTION_CLASS_TO_TYPE_NAME.get(model_function.__class__, None)
        if _type is None:
            raise DReprError("Model function unknown or not supported: %s" % model_function.__class__)
        _yaml['type'] = _type
        
        #TODO implrement
        _yaml['parameter_formatters'] = ""
        _yaml['function_formatter'] = ""
        
        _python_code = inspect.getsource(model_function.func)
        _python_code = textwrap.dedent(_python_code) #remove indentation
        _python_code = _python_code.replace("@staticmethod\n","") #remove @staticmethod decorator
        #TODO what about other decorators?
        _yaml['python_code'] = _python_code
        
        return dict(model_function=_yaml) # wrap inner yaml inside a 'model_function' namespace
    
    def write(self):
        self._yaml = self._make_representation(self._model_function)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml, _h, default_flow_style=False)

class ModelFunctionYamlReader(DReprReaderMixin, ModelFunctionDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'reader'
    FORBIDDEN_TOKENS = ['eval', 'exec', 'execfile', 'file', 'global', 'import', '__import__', 'input', 
                        'nonlocal', 'open', 'reload', 'self', 'super']
    
    def __init__(self, input_io_handle):
        super(ModelFunctionYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_function=None)

    @staticmethod
    def _parse_model_function(input_string):
        _tokens = tokenize.generate_tokens(StringIO.StringIO(input_string).readline)
        for _toknum, _tokval, _spos, _epos, _line_string  in _tokens:
            if _tokval in ModelFunctionYamlReader.FORBIDDEN_TOKENS:
                raise DReprError("Encountered forbidden token '%s' in user-entered code on line '%s'."
                                    % (_tokval, _line_string))
    
        if "__" in input_string:
            raise DReprError("Model function input must not contain '__'!")
    
        __locals_pointer = [None] #TODO better solution?
        input_string += "\n__locals_pointer[0] = __locals()"
        exec(input_string, {"__builtins__":{"__locals":locals}, "__locals_pointer":__locals_pointer})
        _locals = __locals_pointer[0]
        del _locals["__builtins__"]
        del _locals["__locals_pointer"]
        return _locals.values()[0] #TODO adjust for multifits
        
    @staticmethod
    def _make_object(yaml):
        _yaml = yaml["model_function"]

        # -- determine model function class from type
        _model_function_type = _yaml['type']
        _class = ModelFunctionYamlReader._MODEL_FUNCTION_TYPE_NAME_TO_CLASS.get(_model_function_type, None)
        if _class is None:
            raise DReprError("Model function type unknown or not supported: {}".format(_model_function_type))
        _python_function = ModelFunctionYamlReader._parse_model_function(_yaml["python_code"])

        _model_function_object = _class(_python_function)
        
        #TODO construct model function formatter, argument formatters
        
        return _model_function_object
    
    def read(self):
        with self._ihandle as _h:
            self._yaml = yaml.load(_h)
        return self._make_object(self._yaml)

# register the above classes in the module-level dictionary
ModelFunctionYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
