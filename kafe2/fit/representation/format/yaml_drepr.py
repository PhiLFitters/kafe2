from .._base import DReprError
from .._yaml_base import YamlWriterMixin, YamlReaderMixin
from ._base import ModelFunctionFormatterDReprBase, ModelParameterFormatterDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from ....fit._base import ModelParameterFormatter
from ....fit import (HistModelDensityFunctionFormatter, IndexedModelFunctionFormatter,
                     XYModelFunctionFormatter, XYMultiModelFunctionFormatter)
from .._yaml_base import YamlReaderException, YamlWriterException

__all__ = ['ModelFunctionFormatterYamlWriter', 'ModelFunctionFormatterYamlReader', 
           'ModelParameterFormatterYamlWriter', 'ModelParameterFormatterYamlReader']

class ModelFunctionFormatterYamlWriter(YamlWriterMixin, ModelFunctionFormatterDReprBase):
    
    def __init__(self, model_function_formatter, output_io_handle):
        super(ModelFunctionFormatterYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_function_formatter=model_function_formatter)
    
    @classmethod
    def _make_representation(cls, model_function_formatter):
        _yaml_doc = dict()
        _class = model_function_formatter.__class__
        
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Model function formatter unknown or not supported: %s" % _class)
        _yaml_doc['type'] = _type
        
        #TODO should there be a property for _arg_formatters?
        _yaml_doc['arg_formatters'] = [ModelParameterFormatterYamlWriter._make_representation(_arg_formatter) 
                                   for _arg_formatter in model_function_formatter._arg_formatters]
        
        if _class is XYMultiModelFunctionFormatter:
            _singular_formatters_yaml = [
                ModelFunctionFormatterYamlWriter._make_representation(_singular_formatter)
                for _singular_formatter in model_function_formatter._singular_formatters
            ]
            for _singular_formatter_yaml in _singular_formatters_yaml:
                #arg formatters are already stored in the multi formatter
                del _singular_formatter_yaml['arg_formatters']
            _yaml_doc['singular_formatters'] = _singular_formatters_yaml
        else:
            _yaml_doc['name'] = model_function_formatter.name
            _yaml_doc['latex_name'] = model_function_formatter.latex_name
        
            #TODO resolve inconsistent naming
            _yaml_doc['expression_string'] = model_function_formatter.expression_format_string
            _yaml_doc['latex_expression_string'] = model_function_formatter.latex_expression_format_string

            #TODO should there be properties for these calls?
            if _class in (HistModelDensityFunctionFormatter, XYModelFunctionFormatter):
                _yaml_doc['x_name'] = model_function_formatter._x_name
                _yaml_doc['latex_x_name'] = model_function_formatter._latex_x_name
            elif _class is IndexedModelFunctionFormatter:
                _yaml_doc['index_name'] = model_function_formatter._index_name
                _yaml_doc['latex_index_name'] = model_function_formatter._latex_index_name
            else:
                raise YamlWriterException("Unkonwn formatter type")
        
        return _yaml_doc
    
class ModelFunctionFormatterYamlReader(YamlReaderMixin, ModelFunctionFormatterDReprBase):
    
    def __init__(self, input_io_handle):
        super(ModelFunctionFormatterYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_function_formatter=None)

    @classmethod
    def _get_required_keywords(cls, yaml_doc, formatter_class):
        if formatter_class in (HistModelDensityFunctionFormatter, 
                XYModelFunctionFormatter, IndexedModelFunctionFormatter):
            return ['name']
        elif formatter_class is XYMultiModelFunctionFormatter:
            return ['singular_formatters']
        else:
            raise YamlReaderException("Unknown formatter type")

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine model function formatter class from type
        _type = yaml_doc.pop('type')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_type)

        if _class is XYMultiModelFunctionFormatter:
            _constructor_kwargs = dict()
            _singular_formatters_yaml = yaml_doc.pop('singular_formatters')
            if not isinstance(_singular_formatters_yaml, list):
                _singular_formatters_yaml = [_singular_formatters_yaml]
            #TODO validate input
            #TODO implicit sub-object type
            _singular_formatters_list = []
            # don't pop arg_formatters, singular formatters only need a copy,
            # will be popped by the multi formatter
            _arg_formatters = yaml_doc.get('arg_formatters')
            for _singular_formatter_yaml in _singular_formatters_yaml:
                _singular_formatter_yaml['arg_formatters'] = _arg_formatters
                _singular_formatters_list.append(
                    ModelFunctionFormatterYamlReader._make_object(_singular_formatter_yaml)
                )
            _constructor_kwargs['singular_formatters'] = _singular_formatters_list
            
        else:
            _kwarg_list = ['name', 'latex_name', 'expression_string', 'latex_expression_string']
            if _class in (HistModelDensityFunctionFormatter, XYModelFunctionFormatter):
                _kwarg_list.append('x_name')
                _kwarg_list.append('latex_x_name')
            elif _class is IndexedModelFunctionFormatter:
                _kwarg_list.append('index_name')
                _kwarg_list.append('latex_index_name')
            else:
                raise YamlReaderException("Unkonwn formatter type")
            _constructor_kwargs = {key: yaml_doc.pop(key, None) for key in _kwarg_list}
        
        _constructor_kwargs['arg_formatters'] = [ModelParameterFormatterYamlReader._make_object(_representation)
                                                 for _representation in yaml_doc.pop('arg_formatters', [])]
        _model_function_formatter_object = _class(**_constructor_kwargs)
        
        return _model_function_formatter_object, yaml_doc
    
class ModelParameterFormatterYamlWriter(YamlWriterMixin, ModelParameterFormatterDReprBase):
    
    def __init__(self, model_parameter_formatter, output_io_handle):
        super(ModelParameterFormatterYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_parameter_formatter=model_parameter_formatter)
    
    @classmethod
    def _make_representation(cls, model_parameter_formatter):
        _yaml_doc = dict()
        
        _yaml_doc['name'] = model_parameter_formatter.name
        # parameter value and error are not part of the representation
        # because they belong to the parametric model
        #_yaml['value'] = model_parameter_formatter.value
        #_yaml['error'] = model_parameter_formatter.error
        _yaml_doc['latex_name'] = model_parameter_formatter.latex_name

        return _yaml_doc
    
class ModelParameterFormatterYamlReader(YamlReaderMixin, ModelParameterFormatterDReprBase):
    
    def __init__(self, input_io_handle):
        super(ModelParameterFormatterYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_parameter_formatter=None)

    @classmethod
    def _type_required(cls):
        return False

    @classmethod
    def _get_required_keywords(cls, yaml_doc, kafe_object_class):
        return ['name']

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # value and error are not part of the representation
        # because they belong to the parametric model
        _name = yaml_doc.pop('name')
        _latex_name = yaml_doc.pop('latex_name', None)
        _model_parameter_formatter_object = ModelParameterFormatter(name=_name, latex_name=_latex_name)
        
        return _model_parameter_formatter_object, yaml_doc
    
# register the above classes in the module-level dictionary
ModelFunctionFormatterYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionFormatterYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
ModelParameterFormatterYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelParameterFormatterYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)

