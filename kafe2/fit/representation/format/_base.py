import abc
import six

from kafe2.fit.indexed import IndexedModelFunctionFormatter
from kafe2.fit.histogram import HistModelDensityFunctionFormatter
from kafe2.fit.xy import XYModelFunctionFormatter
from kafe2.fit.xy_multi import XYMultiModelFunctionFormatter
from kafe2.fit.representation._base import GenericDReprBase

__all__ = ["ModelFunctionFormatterDReprBase", "ModelParameterFormatterDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class ModelFunctionFormatterDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model_function_formatter'

    #TODO type aliases
    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistModelDensityFunctionFormatter: 'histogram',
        IndexedModelFunctionFormatter: 'indexed',
        XYModelFunctionFormatter: 'xy',
        XYMultiModelFunctionFormatter: 'xy_multi'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistModelDensityFunctionFormatter,
        'indexed': IndexedModelFunctionFormatter,
        'xy': XYModelFunctionFormatter,
        'xy_multi': XYMultiModelFunctionFormatter
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

