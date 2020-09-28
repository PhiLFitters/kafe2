import numpy as np

from ..histogram import HistContainer, HistFit
from ..indexed import IndexedContainer, IndexedFit
from ..unbinned import UnbinnedContainer, UnbinnedFit
from ..xy import XYContainer, XYFit

__all__ = ['Fit']


def Fit(data=None, model_function=None, minimizer=None, **kwargs):
    """A convenience wrapper for simple fit creation. For more control over the fit creation use the corresponding Fit
    classes.

    :param data: A :py:obj:`~kafe2.fit._base.DataContainerBase`-derived data container, containing the data
                           used in the fit.
    :type data: IndexedContainer or HistContainer or UnbinnedContainer or XYContainer or list
    :param model_function: The model function used in the fit.
    :param minimizer: The minimizer backend to use for the fit. This can be ``'scipy'``, ``'iminuit'`` or ``'tminuit'``,
        depending on the installed backends.
    :type minimizer: str or None
    :param kwargs: Any further keyword arguments for the according fit types. For more information, refer to their
        respective documentation.
    """
    container_to_fit = {IndexedContainer: IndexedFit,
                        HistContainer: HistFit,
                        UnbinnedContainer: UnbinnedFit,
                        XYContainer: XYFit,
                        list: XYFit,
                        np.ndarray: XYFit}
    try:
        fit_class = container_to_fit[type(data)]
    except KeyError:
        raise TypeError("Unknown or unsupported data container type {}."
                        "Supported types are {}".format(type(data), container_to_fit.keys()))
    # other errors will raise during creation of the fit object
    if model_function is None:
        return fit_class(data, minimizer=minimizer, **kwargs)  # use default model function
    # model function is not always called model_function e.g. histogram fit, don't use keyword!
    return fit_class(data, model_function, minimizer=minimizer, **kwargs)
