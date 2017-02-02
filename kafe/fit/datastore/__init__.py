import abc


class DataContainerException(Exception):
    pass

class DataContainerBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta