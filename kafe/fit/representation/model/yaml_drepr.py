import inspect
import numpy as np
import StringIO
import textwrap
import tokenize
import yaml

from .._base import DReprError, DReprWriterMixin, DReprReaderMixin
from ..container import DataContainerYamlWriter, DataContainerYamlReader
from ..format import ModelFunctionFormatterYamlWriter, ModelFunctionFormatterYamlReader
from ._base import ModelFunctionDReprBase, ParametricModelDReprBase
from .. import _AVAILABLE_REPRESENTATIONS

__all__ = ['ModelFunctionYamlWriter', 'ModelFunctionYamlReader', 'ParametricModelYamlWriter', 'ParametricModelYamlReader']

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
        
        _yaml.update(ModelFunctionFormatterYamlWriter._make_representation(model_function.formatter))
        
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
        
        #construct model function formatter if specified
        if 'model_function_formatter' in _yaml:
            _model_function_object._formatter = ModelFunctionFormatterYamlReader._make_object(_yaml)
        
        return _model_function_object
    
    def read(self):
        with self._ihandle as _h:
            self._yaml = yaml.load(_h)
        return self._make_object(self._yaml)

class ParametricModelYamlWriter(DReprWriterMixin, ParametricModelDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'writer'

    def __init__(self, parametric_model, output_io_handle):
        super(ParametricModelYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            parametric_model=parametric_model)
    
    @staticmethod
    def _make_representation(parametric_model):
        _yaml = dict()

        # -- determine model function type
        _type = ParametricModelYamlWriter._PARAMETRIC_MODEL_CLASS_TO_TYPE_NAME.get(parametric_model.__class__, None)
        if _type is None:
            raise DReprError("Parametric model unknown or not supported: %s" % parametric_model.__class__)
        _yaml['type'] = _type
        
                # -- write representation for container types
        if _type == 'indexed':
            _yaml['data'] = parametric_model.data.tolist()
        elif _type == 'xy':
            _yaml['x_data'] = parametric_model.x.tolist()
            _yaml['y_data'] = parametric_model.y.tolist()
        elif _type == 'histogram':
            _yaml['bin_edges'] = parametric_model.bin_edges.tolist()
            _yaml['raw_data'] = list(map(float, parametric_model.raw_data))  # float64 -> float
        else:
            raise NotImplemented("Container type unknown or not supported: {}".format(_type))

        _parameters = parametric_model.parameters
        if isinstance(_parameters, np.ndarray):
            _parameters = _parameters.tolist() #better readability in file
        _yaml['model_parameters'] = _parameters

        # -- write error representation for all container types
        if parametric_model.has_errors:
            DataContainerYamlWriter._write_errors_to_yaml(parametric_model, _yaml)
        
        _yaml.update(ModelFunctionYamlWriter._make_representation(parametric_model._model_function_object))
        
        return dict(parametric_model=_yaml) # wrap inner yaml inside a 'parametric_model' namespace
    
    def write(self):
        self._yaml = self._make_representation(self._parametric_model)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml, _h, default_flow_style=False)

class ParametricModelYamlReader(DReprReaderMixin, ParametricModelDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'reader'
    
    def __init__(self, input_io_handle):
        super(ParametricModelYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            parametric_model=None)

        
    @staticmethod
    def _make_object(yaml):
        _yaml = yaml["parametric_model"]

        # -- determine model function class from type
        _parametric_model_type = _yaml['type']
        _class = ParametricModelYamlReader._PARAMETRIC_MODEL_TYPE_NAME_TO_CLASS.get(_parametric_model_type, None)
        if _class is None:
            raise DReprError("Model function type unknown or not supported: {}".format(_parametric_model_type))
        
        _kwarg_list = ['model_parameters']
        if _parametric_model_type == 'histogram':
            _kwarg_list.append('n_bins')
            _kwarg_list.append('bin_range')
            _kwarg_list.append('bin_edges')
            #TODO implement parsing
            _kwarg_list.append('model_density_func_antiderivative')
        elif _parametric_model_type == 'indexed':
            _kwarg_list.append('shape_like')
        elif _parametric_model_type == 'xy':
            _kwarg_list.append('x_data')
        _constructor_kwargs = {key: _yaml.get(key, None) for key in _kwarg_list}
        if _parametric_model_type is 'histogram':
            _constructor_kwargs['model_density_func'] = ModelFunctionYamlReader._make_object(_yaml)
        elif _parametric_model_type in ('indexed', 'xy'):
            _constructor_kwargs['model_func'] = ModelFunctionYamlReader._make_object(_yaml)
        _parametric_model_object = _class(**_constructor_kwargs)
        
        # -- process error sources
        if _parametric_model_type == 'xy':
            _xerrs = _yaml.get('x_errors', [])
            _yerrs = _yaml.get('y_errors', [])
            _errs = _xerrs + _yerrs
            _axes = [0] * len(_xerrs) + [1] * len(_yerrs)  # 0 for 'x', 1 for 'y'
        else:
            _errs = _yaml.get('errors', [])
            _axes = [None] * len(_errs)

        # add error sources, if any
        for _err, _axis in zip(_errs, _axes):
            _add_kwargs = dict()
            # translate and check that all required keys are present
            try:
                _err_type = _err['type']

                _add_kwargs['name'] = _err['name']

                if _err_type == 'simple':
                    _add_kwargs['err_val']= _err['error_value']
                    _add_kwargs['correlation']= _err['correlation_coefficient']
                elif _err_type == 'matrix':
                    _add_kwargs['err_matrix'] = _err['matrix']
                    _add_kwargs['matrix_type'] = _err['matrix_type']
                    _add_kwargs['err_val'] = _err.get('error_value', None)  # only mandatory for cor mats; check done later
                else:
                    raise DReprError("Unknown error type '{}'. "
                                     "Valid: {}".format(_err_type, ('simple', 'matrix')))

                _add_kwargs['relative'] = _err['relative']

                # if needed, specify the axis (only for 'xy' containers)
                if _axis is not None:
                    _add_kwargs['axis'] = _axis
            except KeyError as e:
                # KeyErrors mean the YAML is incomplete -> raise
                raise DReprError("Missing required key '%s' for error specification" % e.args[0])

            # add error to parametric model
            DataContainerYamlReader._add_error_to_container(_err_type, _parametric_model_object, **_add_kwargs)
        
        return _parametric_model_object
    
    def read(self):
        with self._ihandle as _h:
            self._yaml = yaml.load(_h)
        return self._make_object(self._yaml)

# register the above classes in the module-level dictionary
ModelFunctionYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
ParametricModelYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ParametricModelYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
