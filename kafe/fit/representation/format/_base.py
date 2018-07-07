import abc

from kafe.fit.indexed import IndexedModelFunctionFormatter
from kafe.fit.xy import XYModelFunctionFormatter
from kafe.fit.histogram import HistModelDensityFunctionFormatter
from kafe.fit.representation._base import GenericDReprBase

__all__ = ["ModelFunctionDReprBase", "ParametricModelDReprBase"]


class ModelFunctionFormatterDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = 'model_function_formatter'

    _MODEL_FUNCTION_FORMATTER_CLASS_TO_TYPE_NAME = {
        IndexedModelFunctionFormatter: 'indexed',
        XYModelFunctionFormatter: 'xy',
        HistModelDensityFunctionFormatter: 'histogram'
    }
    _MODEL_FUNCTION_FORMATTER_TYPE_NAME_TO_CLASS = {
        'indexed': IndexedModelFunctionFormatter,
        'xy': XYModelFunctionFormatter,
        'histogram': HistModelDensityFunctionFormatter
    }

    def __init__(self, model_function=None):
        self._model_function = model_function
        super(ModelFunctionFormatterDReprBase, self).__init__()

class ModelParameterDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = 'model_parameter_formatter'

    def __init__(self, model=None):
        self._model = model
        super(ModelParameterDReprBase, self).__init__()

