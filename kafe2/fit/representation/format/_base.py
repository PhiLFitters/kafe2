import abc

# import formatter classes
from ..._base.format import ModelFunctionFormatter
from ...indexed import IndexedModelFunctionFormatter
from .._base import GenericDReprBase

__all__ = ["ModelFunctionFormatterDReprBase", "ParameterFormatterDReprBase"]


class ModelFunctionFormatterDReprBase(GenericDReprBase, metaclass=abc.ABCMeta):
    BASE_OBJECT_TYPE_NAME = "model_function_formatter"

    _CLASS_TO_OBJECT_TYPE_NAME = {
        ModelFunctionFormatter: "base",
        IndexedModelFunctionFormatter: "indexed",
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        "base": ModelFunctionFormatter,
        "histogram": ModelFunctionFormatter,
        "indexed": IndexedModelFunctionFormatter,
        "unbinned": ModelFunctionFormatter,
        "xy": ModelFunctionFormatter,
    }

    def __init__(self, model_function_formatter=None):
        self._kafe_object = model_function_formatter
        super(ModelFunctionFormatterDReprBase, self).__init__()


class ParameterFormatterDReprBase(GenericDReprBase, metaclass=abc.ABCMeta):
    BASE_OBJECT_TYPE_NAME = "parameter_formatter"

    def __init__(self, model_parameter_formatter=None):
        self._kafe_object = model_parameter_formatter
        super(ParameterFormatterDReprBase, self).__init__()
