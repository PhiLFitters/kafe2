import abc
import six

from .._base import GenericDReprBase
# import formatter classes
from ..._base.format import ModelFunctionFormatter
from ...indexed import IndexedModelFunctionFormatter

__all__ = ["ModelFunctionFormatterDReprBase", "ModelParameterFormatterDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class ModelFunctionFormatterDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model_function_formatter'

    # This dict is currently not used. Because there are only two different formatters adn most fits use the base format
    # Case separation is currently only handled in the yaml_drepr.py
    _CLASS_TO_OBJECT_TYPE_NAME = {
        ModelFunctionFormatter: 'base',
        IndexedModelFunctionFormatter: 'indexed'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'base': ModelFunctionFormatter,
        'indexed': IndexedModelFunctionFormatter
    }

    def __init__(self, model_function_formatter=None):
        self._kafe_object = model_function_formatter
        super(ModelFunctionFormatterDReprBase, self).__init__()


@six.add_metaclass(abc.ABCMeta)
class ModelParameterFormatterDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model_parameter_formatter'

    def __init__(self, model_parameter_formatter=None):
        self._kafe_object = model_parameter_formatter
        super(ModelParameterFormatterDReprBase, self).__init__()
