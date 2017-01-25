"""

"""

from collections import OrderedDict
from ast import parse


class ParameterException(Exception):
    pass


class ParameterSpaceException(Exception):
    pass



class Parameter(object):

    RESERVED_PARAMETER_NAMES = ('__all__', '__real__')

    def __init__(self, name, value):
        self.name = name
        self.value = value

    @staticmethod
    def _check_parameter_name_raise(parameter_name):
        if parameter_name in Parameter.RESERVED_PARAMETER_NAMES:
            raise ParameterException("Invalid parameter name: '%s' is a reserved keyword!"
                                     % (parameter_name,))
        try:
            parse('dict(%s=0.123)' % (parameter_name,))
        except SyntaxError:
            raise ParameterException("Invalid parameter name '%s'. Must be Python identifier!"
                                     % (parameter_name,))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._check_parameter_name_raise(name)
        self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class ParameterSpace(object):
    """Manages `Parameter` aliasing and constructing value vectors/error vectors/error matrices for arbitrary subsets
    of the parameter space."""

    def __init__(self):
        self._all_pars = OrderedDict()  # keys are parameter ids, values are `Parameter` objects

        # "Nexus": metadata about parameter aliases
        self.__name_id_map = OrderedDict()  # map
        self.__dim = 0
        self.__real_pars = []  # parameter names which are not aliases
        self._q_nexus_stale = True

        self._covariances = None  # this is to be populated by a `Fit` object


    # -- private methods

    @staticmethod
    def _check_parameter_name_invalid_reserved_raise(parameter_name):
        try:
            parse('dict(%s=0.123)' % (parameter_name,))
        except SyntaxError:
            raise ParameterException("Invalid parameter name '%s'. Must be Python identifier!"
                                     % (parameter_name,))

    def _rebuild_nexus(self):
        _seen = []
        _ids = {}
        _dim = 0
        for _pn, _p in self._all_pars.iteritems():
            if _p not in _seen:
                _seen.append(_p)
                _dim += 1
            _ids[_pn] = _seen.index(_p)

        self.__name_id_map = _ids
        self.__dim = _dim
        self.__real_pars = _seen
        self._q_nexus_stale = False

    # -- private "properties"

    @property
    def _name_id_map(self):
        if self._q_nexus_stale:
            self._rebuild_nexus()
        return self.__name_id_map

    @property
    def _real_pars(self):
        if self._q_nexus_stale:
            self._rebuild_nexus()
        return self.__real_pars

    @property
    def _dim(self):
        if self._q_nexus_stale:
            self._rebuild_nexus()
        return self.__dim

    # -- public interface

    def _get_one(self, name, default):
        return self._all_pars.get(name, default)

    def get(self, name=None, default=None):
        if name is None or name == "__all__":
            # return all parameters
            return [self._get_one(_n, default) for _n in self._all_pars.keys()]
        elif name == "__real__":
            # return all parameters that are not aliases
            return self._real_pars
        elif isinstance(name, str) or isinstance(name, unicode):
            # return one parameter
            return self._get_one(name, default)
        else:
            # return a list of parameters
            try:
                iter(name)
            except:
                raise ParameterSpaceException("Getting parameter(s) with name '%r' failed: not string or iterable"
                                              "of string!" % (name,))
            else:
                return [self.get(_n, default) for _n in name]

    def get_values(self, name=None, default=None):
        _l = self.get(name, default)
        if isinstance(_l, list):
            return [_p.value if _p is not None else None for _p in _l]
        else:
            return _l.value if _l is not None else None

    def set(self, name, value, create=False):
        _p = self._all_pars.get(name)
        if _p is None and create == False:
            raise ParameterSpaceException("Cannot create parameter '%s': not allowed!" % (name,))
        elif _p is not None and create == True:
            raise ParameterSpaceException("Cannot create parameter '%s': exists!" % (name,))
        elif _p is None and create == True:
            try:
                self._all_pars[name] = Parameter(name, value)
            except ParameterException as pe:
                # re-raise ParameterException as ParameterSpaceException
                raise ParameterSpaceException(pe.message)
            self._q_nexus_stale = True
        else:
            _p.value = value

    def new_alias(self, alias, name):
        _p = self._all_pars.get(name)
        if _p is None:
            raise ParameterSpaceException("Cannot create alias of non-existent parameter '%s'!" % (name,))

        _pa = self._all_pars.get(alias)
        if _pa is not None:
            raise ParameterSpaceException("Cannot create alias '%s': exists!" % (alias,))

        try:
            Parameter._check_parameter_name_raise(alias)
            self._all_pars[alias] = _p
        except ParameterException as pe:
            # re-raise ParameterException as ParameterSpaceException
            raise ParameterSpaceException(pe.message)

        self._q_nexus_stale = True

    @property
    def ids(self):
        if self._q_nexus_stale:
            self._rebuild_nexus()
        return self._name_id_map

    @property
    def dimension(self):
        if self._q_nexus_stale:
            self._rebuild_nexus()
        return self._dim


# class ParameterDependencyGraph(object):
#     """Implements the observer model for `Parameter` objects in a `ParameterSpace`."""
#     def __init__(self):
#         self._graph = dict()
#
#     def add_dependency(self):
#         pass
#
#     def has_cycle(self):
#         """Check for cyclic dependencies and warn."""
#         raise NotImplementedError