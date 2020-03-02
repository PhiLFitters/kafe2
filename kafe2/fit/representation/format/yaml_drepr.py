from .._base import DReprError
from .._yaml_base import YamlWriterMixin, YamlReaderMixin
from ._base import ModelFunctionFormatterDReprBase, ModelParameterFormatterDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from ..._base import ParameterFormatter, ModelFunctionFormatter
from ...indexed import IndexedModelFunctionFormatter
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
        """Create a representation of a :py:obj:`ModelFunctionFormatter` object as a dictionary.

        :param model_function_formatter: The :py:obj:`ModelFunctionFormatter` object to represent.
        :type model_function_formatter: ModelFunctionFormatter
        :return: Dictionary containing all information about the :py:obj:`ModelFunctionFormatter` object.
        """
        _yaml_doc = dict()
        _class = model_function_formatter.__class__
        
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Model function formatter unknown or not supported: %s" % _class)
        
        #TODO should there be a property for _arg_formatters?
        _yaml_doc['arg_formatters'] = [ModelParameterFormatterYamlWriter._make_representation(_arg_formatter)
                                       for _arg_formatter in model_function_formatter._arg_formatters]

        _yaml_doc['name'] = model_function_formatter.name
        _yaml_doc['latex_name'] = model_function_formatter.latex_name

        #TODO resolve inconsistent naming
        _yaml_doc['expression_string'] = model_function_formatter.expression_format_string
        _yaml_doc['latex_expression_string'] = model_function_formatter.latex_expression_format_string

        # TODO should there be properties for these calls?
        if _class is IndexedModelFunctionFormatter:
            _yaml_doc['index_name'] = model_function_formatter.index_name
            _yaml_doc['latex_index_name'] = model_function_formatter.latex_index_name
        elif _class is ModelFunctionFormatter:
            pass
        else:
            raise YamlWriterException("Unknown formatter type!")

        return _yaml_doc


class ModelFunctionFormatterYamlReader(YamlReaderMixin, ModelFunctionFormatterDReprBase):
    
    def __init__(self, input_io_handle):
        super(ModelFunctionFormatterYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_function_formatter=None)

    @classmethod
    def _type_required(cls):
        return False

    @classmethod
    def _get_required_keywords(cls, yaml_doc, formatter_class):
        return ['name']

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine model function formatter class (only indexed and base)
        _type = 'indexed' if 'index_name' in yaml_doc else 'base'
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_type)

        _kwarg_list = ['name', 'latex_name', 'expression_string', 'latex_expression_string']
        if _class is IndexedModelFunctionFormatter:
            _kwarg_list.append('index_name')
            _kwarg_list.append('latex_index_name')
        elif _class is ModelFunctionFormatter:
            pass
        else:
            raise YamlReaderException("Unknown formatter type!")
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
        # _yaml['value'] = model_parameter_formatter.value
        # _yaml['error'] = model_parameter_formatter.error
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
        _model_parameter_formatter_object = ParameterFormatter(name=_name, latex_name=_latex_name)
        
        return _model_parameter_formatter_object, yaml_doc


# register the above classes in the module-level dictionary
ModelFunctionFormatterYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionFormatterYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
ModelParameterFormatterYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelParameterFormatterYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
