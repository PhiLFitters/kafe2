import abc
import six


__all__ = ["FitEnsembleBase", "FitEnsembleException"]


class FitEnsembleException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class FitEnsembleBase(object):
    """
    Object for generating ensembles of fits to pseudo-data generated according to the
    specified uncertainty model.

    This is a purely abstract class implementing the minimal interface required by all
    types of fit ensembles.
    """

    FIT_TYPE = None
