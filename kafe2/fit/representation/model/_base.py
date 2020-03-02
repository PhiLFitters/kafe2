import abc
import six

from .._base import GenericDReprBase
# import ModelFunction classes
from ..._base import ModelFunctionBase
from ...histogram import HistModelFunction, HistParametricModel
from ...indexed import IndexedModelFunction, IndexedParametricModel
from ...unbinned import UnbinnedParametricModel
from ...xy import XYParametricModel

__all__ = ["ModelFunctionDReprBase", "ParametricModelDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class ModelFunctionDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model_function'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistModelFunction: 'histogram',
        IndexedModelFunction: 'indexed',
        ModelFunctionBase: 'base'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistModelFunction,
        'indexed': IndexedModelFunction,
        'unbinned': ModelFunctionBase,
        'xy': ModelFunctionBase,  # type from fit is passed to model function, needs to be resolved
        'base': ModelFunctionBase
    }

    def __init__(self, model_function=None):
        self._kafe_object = model_function
        super(ModelFunctionDReprBase, self).__init__()


@six.add_metaclass(abc.ABCMeta)
class ParametricModelDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistParametricModel: 'histogram',
        IndexedParametricModel: 'indexed',
        UnbinnedParametricModel: 'unbinned',
        XYParametricModel: 'xy'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistParametricModel,
        'indexed': IndexedParametricModel,
        'unbinned': UnbinnedParametricModel,
        'xy': XYParametricModel
    }

    def __init__(self, parametric_model=None):
        self._kafe_object = parametric_model
        super(ParametricModelDReprBase, self).__init__()
