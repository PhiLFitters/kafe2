"""

User API: wraps kafe.core functionality

"""

# public interface of submodule 'kafe.fit'

from .indexed import IndexedContainer, IndexedParametricModel, IndexedFit, IndexedPlot
from .histogram import HistContainer, HistParametricModel, HistFit, HistPlot
from .xy import XYContainer, XYParametricModel, XYFit, XYPlot


__all__ = [
    "IndexedContainer", "IndexedParametricModel", "IndexedFit", "IndexedPlot",
    "HistContainer", "HistParametricModel", "HistFit", "HistPlot",
    "XYContainer", "XYParametricModel", "XYFit", "XYPlot"
]

