import abc


class FitEnsembleException(Exception):
    pass


class FitEnsembleBase(object):
    """
    Object for generating ensembles of fits to pseudo-data generated according to the
    specified uncertainty model.

    This is a purely abstract class implementing the minimal interface required by all
    types of fit ensembles.
    """
    __metaclass__ = abc.ABCMeta

    FIT_TYPE = None