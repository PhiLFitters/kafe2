import abc

from kafe.fit.histogram import HistModelFunction, HistParametricModel
from kafe.fit.indexed import IndexedModelFunction, IndexedParametricModel
from kafe.fit.xy import XYModelFunction, XYParametricModel
from kafe.fit.xy_multi import XYMultiModelFunction, XYMultiParametricModel
from kafe.fit.representation._base import GenericDReprBase

__all__ = ["ModelFunctionDReprBase", "ParametricModelDReprBase"]


class ModelFunctionDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = 'model_function'

    _MODEL_FUNCTION_CLASS_TO_TYPE_NAME = {
        HistModelFunction: 'histogram',
        IndexedModelFunction: 'indexed',
        XYModelFunction: 'xy',
        XYMultiModelFunction: 'xy_multi'
    }
    _MODEL_FUNCTION_TYPE_NAME_TO_CLASS = {
        'histogram': HistModelFunction,
        'indexed': IndexedModelFunction,
        'xy': XYModelFunction,
        'xy_multi': XYMultiModelFunction
    }

    def __init__(self, model_function=None):
        self._model_function = model_function
        super(ModelFunctionDReprBase, self).__init__()

class ParametricModelDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = 'model'

    _PARAMETRIC_MODEL_CLASS_TO_TYPE_NAME = {
        IndexedParametricModel: 'indexed',
        XYParametricModel: 'xy',
        HistParametricModel: 'histogram'
    }
    _PARAMETRIC_MODEL_TYPE_NAME_TO_CLASS = {
        'indexed': IndexedParametricModel,
        'xy': XYParametricModel,
        'histogram': HistParametricModel
    }

    def __init__(self, parametric_model=None):
        self._parametric_model = parametric_model
        super(ParametricModelDReprBase, self).__init__()

