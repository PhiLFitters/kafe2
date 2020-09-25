import abc
import six

from .._base import GenericDReprBase
# import formatter classes
from ..._base.format import ModelFunctionFormatter
from ...indexed import IndexedModelFunctionFormatter

__all__ = ["ModelFunctionFormatterDReprBase", "ParameterFormatterDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class ModelFunctionFormatterDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model_function_formatter'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        ModelFunctionFormatter: 'base',
        IndexedModelFunctionFormatter: 'indexed'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'base': ModelFunctionFormatter,
        'histogram': ModelFunctionFormatter,
        'indexed': IndexedModelFunctionFormatter,
        'unbinned': ModelFunctionFormatter,
        'xy': ModelFunctionFormatter,
    }

    def __init__(self, model_function_formatter=None):
        self._kafe_object = model_function_formatter
        super(ModelFunctionFormatterDReprBase, self).__init__()


@six.add_metaclass(abc.ABCMeta)
class ParameterFormatterDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'parameter_formatter'

    def __init__(self, model_parameter_formatter=None):
        self._kafe_object = model_parameter_formatter
        super(ParameterFormatterDReprBase, self).__init__()
