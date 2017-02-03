"""

User API: wraps kafe.core functionality

"""

# public interface of submodule 'kafe.fit'

from .datastore import (HistContainer, HistParametricModel,
                        IndexedContainer, IndexedParametricModel,
                        XYContainer, XYParametricModel)

__all__ = ['HistContainer', 'HistParametricModel',
           'IndexedContainer', 'IndexedParametricModel',
           'XYContainer', 'XYParametricModel']
