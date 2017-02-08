"""

User API: wraps kafe.core functionality

"""

# public interface of submodule 'kafe.fit'

from .containers import (HistContainer, HistParametricModel,
                         IndexedContainer, IndexedParametricModel,
                         XYContainer, XYParametricModel)

from .fitters import (HistFit, IndexedFit, XYFit)

__all__ = [
    'HistContainer', 'HistParametricModel',
    'IndexedContainer', 'IndexedParametricModel',
    'XYContainer', 'XYParametricModel',
    'HistFit', 'IndexedFit', 'XYFit'
]

