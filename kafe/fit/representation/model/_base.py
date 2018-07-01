import abc

from kafe.fit.indexed import IndexedModelFunction, IndexedParametricModel
from kafe.fit.xy import XYModelFunction, XYParametricModel
from kafe.fit.histogram import HistModelFunction, HistParametricModel
from kafe.fit.representation._base import GenericDReprBase

__all__ = ["ModelFunctionDReprBase", "ParametricModelDReprBase"]


class ModelFunctionDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = 'modelfunction'

    _CONTAINER_CLASS_TO_TYPE_NAME = {
        IndexedModelFunction: 'indexed',
        XYModelFunction: 'xy',
        HistModelFunction: 'histogram'
    }
    _CONTAINER_TYPE_NAME_TO_CLASS = {
        'indexed': IndexedModelFunction,
        'xy': XYModelFunction,
        'histogram': HistModelFunction
    }

    def __init__(self, model_function=None):
        self._model_function = model_function
        super(ModelFunctionDReprBase, self).__init__()

class ParametricModelDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = 'model'

    _CONTAINER_CLASS_TO_TYPE_NAME = {
        IndexedParametricModel: 'indexed',
        XYParametricModel: 'xy',
        HistParametricModel: 'histogram'
    }
    _CONTAINER_TYPE_NAME_TO_CLASS = {
        'indexed': IndexedParametricModel,
        'xy': XYParametricModel,
        'histogram': HistParametricModel
    }

    def __init__(self, model=None):
        self._model = model
        super(ParametricModelDReprBase, self).__init__()

