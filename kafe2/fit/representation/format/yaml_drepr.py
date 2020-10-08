from ._base import ModelFunctionFormatterDReprBase, ParameterFormatterDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from .._base import DReprError
from .._yaml_base import YamlReaderException, YamlReaderMixin, YamlWriterException, YamlWriterMixin
from ..._base import ModelFunctionFormatter, ParameterFormatter
from ...indexed import IndexedModelFunctionFormatter

__all__ = ["ModelFunctionFormatterYamlWriter", "ModelFunctionFormatterYamlReader",
           "ParameterFormatterYamlWriter", "ParameterFormatterYamlReader"]


class ModelFunctionFormatterYamlWriter(YamlWriterMixin, ModelFunctionFormatterDReprBase):

    def __init__(self, model_function_formatter, output_io_handle):
        super(ModelFunctionFormatterYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_function_formatter=model_function_formatter)

    @classmethod
    def _make_representation(cls, model_function_formatter):
        """Create a representation of a :py:obj:`ModelFunctionFormatter` object as a dictionary.

        :param model_function_formatter: The :py:obj:`ModelFunctionFormatter` object to represent.
        :type model_function_formatter: ModelFunctionFormatter | IndexedModelFunctionFormatter
        :return: Dictionary containing all information about the :py:obj:`ModelFunctionFormatter`
            object.
        """
        _yaml_doc = dict()
        _class = model_function_formatter.__class__

        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Model function formatter unknown or not supported: %s" % _class)
        _yaml_doc['type'] = _type

        if _class is IndexedModelFunctionFormatter:
            _yaml_doc['index_name'] = model_function_formatter.index_name
            _yaml_doc['latex_index_name'] = model_function_formatter.latex_index_name
        elif _class is ModelFunctionFormatter:
            pass
        else:
            raise YamlWriterException("Unknown formatter type!")

        _yaml_doc['name'] = model_function_formatter.name
        _yaml_doc['latex_name'] = model_function_formatter.latex_name

        _yaml_doc['expression_string'] = model_function_formatter.expression_format_string
        _yaml_doc['latex_expression_string'] = \
            model_function_formatter.latex_expression_format_string

        _arg_formatters_dict = dict()
        for _arg_formatter in model_function_formatter.arg_formatters:
            _arg_formatters_dict[_arg_formatter.name] = _arg_formatter.latex_name
        _yaml_doc['arg_formatters'] = _arg_formatters_dict

        # This is needed when saving a formatter without a model function object in order to
        # know the correct order of the arguments. This is because writing and then reading a yaml
        # file in Py2 will change the order of a dict or OrderedDict. This is not necessary for Py3!
        _yaml_doc['signature'] = [_arg_formatter.name for _arg_formatter in
                                  model_function_formatter.arg_formatters]
        return _yaml_doc


class ModelFunctionFormatterYamlReader(YamlReaderMixin, ModelFunctionFormatterDReprBase):

    def __init__(self, input_io_handle):
        super(ModelFunctionFormatterYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_function_formatter=None)

    @classmethod
    def _modify_yaml_doc(cls, yaml_doc, kafe_object_class, name=None, signature=None):
        # only update keys in yaml doc if they don't exist
        # needed for setting name and signature when creating from a parametric model
        if 'name' not in yaml_doc and name is not None:
            yaml_doc['name'] = name
        if 'signature' not in yaml_doc and signature is not None:
            yaml_doc['signature'] = signature
        return yaml_doc

    @classmethod
    def _get_required_keywords(cls, yaml_doc, formatter_class):
        return ['name', 'signature']

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine model function formatter class (only indexed and base)
        _type = yaml_doc.pop('type')
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

        _signature = yaml_doc.pop('signature')  # either list or dict, definitely a dict when
        # this is leftover from creating a model function
        _arg_formatters_yaml = yaml_doc.pop('arg_formatters', {})
        _arg_formatters = []
        for arg_name in _signature:
            _representation = _arg_formatters_yaml.pop(arg_name, {})
            if isinstance(_representation, str):
                _representation = {'latex_name': _representation}
            _representation.update({'id': arg_name})
            _arg_formatters.append(ParameterFormatterYamlReader._make_object(_representation))
        _constructor_kwargs['arg_formatters'] = _arg_formatters

        _model_function_formatter_object = _class(**_constructor_kwargs)

        return _model_function_formatter_object, yaml_doc


class ParameterFormatterYamlWriter(YamlWriterMixin, ParameterFormatterDReprBase):

    def __init__(self, model_parameter_formatter, output_io_handle):
        super(ParameterFormatterYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_parameter_formatter=model_parameter_formatter)

    @classmethod
    def _make_representation(cls, model_parameter_formatter):
        _yaml_doc = dict()
        _yaml_doc['id'] = model_parameter_formatter.arg_name
        _yaml_doc['name'] = model_parameter_formatter.name
        # parameter value and error are not part of the representation
        # because they belong to the parametric model
        _yaml_doc['latex_name'] = model_parameter_formatter.latex_name

        return _yaml_doc


class ParameterFormatterYamlReader(YamlReaderMixin, ParameterFormatterDReprBase):

    def __init__(self, input_io_handle):
        super(ParameterFormatterYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_parameter_formatter=None)

    @classmethod
    def _type_required(cls):
        return False

    @classmethod
    def _get_required_keywords(cls, yaml_doc, kafe_object_class):
        return ['id']

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # value and error are not part of the representation
        # because they belong to the parametric model
        _id = yaml_doc.pop('id')
        _name = yaml_doc.pop('name', None)
        _latex_name = yaml_doc.pop('latex_name', None)
        _model_parameter_formatter_object = ParameterFormatter(_id, name=_name,
                                                               latex_name=_latex_name)
        return _model_parameter_formatter_object, yaml_doc


# register the above classes in the module-level dictionary
ModelFunctionFormatterYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionFormatterYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
ParameterFormatterYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ParameterFormatterYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
