import yaml

from .._base import DReprError, DReprWriterMixin, DReprReaderMixin
from ._base import ModelFunctionFormatterDReprBase, ModelParameterFormatterDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from kafe.fit._base import ModelParameterFormatter
from kafe.fit.xy.format import XYModelFunctionFormatter
from kafe.fit.histogram.format import HistModelDensityFunctionFormatter
from kafe.fit.indexed.format import IndexedModelFunctionFormatter

__all__ = ['ModelFunctionFormatterYamlWriter', 'ModelFunctionFormatterYamlReader', 
           'ModelParameterFormatterYamlWriter', 'ModelParameterFormatterYamlReader']

class ModelFunctionFormatterYamlWriter(DReprWriterMixin, ModelFunctionFormatterDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'writer'
    
    def __init__(self, model_function_formatter, output_io_handle):
        super(ModelFunctionFormatterYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_function_formatter=model_function_formatter)
    
    @staticmethod
    def _make_representation(model_function_formatter):
        _yaml = dict()
        
        _type = ModelFunctionFormatterYamlWriter._MODEL_FUNCTION_FORMATTER_CLASS_TO_TYPE_NAME.get(model_function_formatter.__class__, None)
        if _type is None:
            raise DReprError("Model function formatter unknown or not supported: %s" % model_function_formatter.__class__)
        _yaml['type'] = _type
        
        _yaml['name'] = model_function_formatter.name
        _yaml['latex_name'] = model_function_formatter.latex_name
        
        #TODO should there be a property for _arg_formatters?
        _yaml['arg_formatters'] = [ModelParameterFormatterYamlWriter._make_representation(_arg_formatter) 
                                   for _arg_formatter in model_function_formatter._arg_formatters]
        
        #TODO resolve inconsistent naming
        _yaml['expression_string'] = model_function_formatter.expression_format_string
        _yaml['latex_expression_string'] = model_function_formatter.latex_expression_format_string

        #TODO should there be properties for these calls?
        if _type in ('hist', 'xy'):
            _yaml['x_name'] = model_function_formatter._x_name
            _yaml['latex_x_name'] = model_function_formatter._latex_x_name
        if _type is 'indexed':
            _yaml['index_name'] = model_function_formatter._x_name
            _yaml['latex_index_name'] = model_function_formatter._latex_x_name
        
        return dict(model_function_formatter=_yaml) # wrap inner yaml inside a 'model_function_formatter' namespace
    
    def write(self):
        self._yaml = self._make_representation(self._model_function_formatter)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml, _h, default_flow_style=False)

class ModelFunctionFormatterYamlReader(DReprReaderMixin, ModelFunctionFormatterDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'reader'
    
    def __init__(self, input_io_handle):
        super(ModelFunctionFormatterYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_function_formatter=None)

    @staticmethod
    def _make_object(yaml):
        _yaml = yaml["model_function_formatter"]

        # -- determine model function formatter class from type
        _model_function_formatter_type = _yaml['type']
        _class = ModelFunctionFormatterYamlReader._MODEL_FUNCTION_FORMATTER_TYPE_NAME_TO_CLASS.get(_model_function_formatter_type, None)
        if _class is None:
            raise DReprError("Model function formatter type unknown or not supported: {}".format(_model_function_formatter_type))

        _kwarg_list = ['name', 'latex_name', 'expression_string', 'latex_expression_string']
        if _class in (HistModelDensityFunctionFormatter, XYModelFunctionFormatter):
            _kwarg_list.append('x_name')
            _kwarg_list.append('latex_x_name')
        if _class is IndexedModelFunctionFormatter:
            _kwarg_list.append('index_name')
            _kwarg_list.append('latex_index_name')
        _constructor_kwargs = {key: _yaml.get(key, None) for key in _kwarg_list}
        
        _constructor_kwargs['arg_formatters'] = [ModelParameterFormatterYamlReader._make_object(_representation)
                                                 for _representation in _yaml.get('arg_formatters', [])]
        
        _model_function_formatter_object = _class(**_constructor_kwargs)
        
        return _model_function_formatter_object
    
    def read(self):
        with self._ihandle as _h:
            self._yaml = yaml.load(_h)
        return self._make_object(self._yaml)

class ModelParameterFormatterYamlWriter(DReprWriterMixin, ModelParameterFormatterDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'writer'
    
    def __init__(self, model_parameter_formatter, output_io_handle):
        super(ModelParameterFormatterYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_parameter_formatter=model_parameter_formatter)
    
    @staticmethod
    def _make_representation(model_parameter_formatter):
        _yaml = dict()
        
        _yaml['name'] = model_parameter_formatter.name
        # parameter value and error are not part of the representation
        # because they belong to the parametric model
        #_yaml['value'] = model_parameter_formatter.value
        #_yaml['error'] = model_parameter_formatter.error
        _yaml['latex_name'] = model_parameter_formatter.latex_name

        return dict(model_parameter_formatter=_yaml) # wrap inner yaml inside a 'model_parameter_formatter' namespace
    
    def write(self):
        self._yaml = self._make_representation(self._model_parameter_formatter)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml, _h, default_flow_style=False)

class ModelParameterFormatterYamlReader(DReprReaderMixin, ModelParameterFormatterDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'reader'
    
    def __init__(self, input_io_handle):
        super(ModelParameterFormatterYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_parameter_formatter=None)

    @staticmethod
    def _make_object(yaml):
        _yaml = yaml["model_parameter_formatter"]
        
        # value and error are not part of the representation
        # because they belong to the parametric model
        _kwarg_list = ['name', 'latex_name']
        _arg_dict = {key: _yaml.get(key, None) for key in _kwarg_list}
        _model_parameter_formatter_object = ModelParameterFormatter(**_arg_dict)
        
        return _model_parameter_formatter_object
    
    def read(self):
        with self._ihandle as _h:
            self._yaml = yaml.load(_h)
        return self._make_object(self._yaml)

# register the above classes in the module-level dictionary
ModelFunctionFormatterYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionFormatterYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
ModelParameterFormatterYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelParameterFormatterYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)

