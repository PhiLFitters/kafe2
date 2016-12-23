"""

"""

from collections import OrderedDict

class Parameter:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class ParameterSpace:
    def __init__(self):
        self._all_pars = OrderedDict()

        self._name_id_map = OrderedDict()
        self._real_pars = []
        self._dim = 0

        self._par_cov_mat = NotImplemented

        self._q_nexus_stale = True

    # -- private methods

    def _rebuild_nexus(self):
        _seen = []
        _ids = {}
        _dim = 0
        for _pn, _p in self._all_pars.iteritems():
            if _p not in _seen:
                _seen.append(_p)
                _dim += 1
            _ids[_pn] = _seen.index(_p)

        self._name_id_map = _ids
        self._dim = _dim
        self._real_pars = _seen
        self._q_nexus_stale = False

    # -- public interface

    def get(self, name, default=None):
        return self._all_pars.get(name, default)

    def set(self, name, value, create=False):
        _p = self._all_pars.get(name)
        if _p is None and create==False:
            raise ParameterSpaceException("Cannot create parameter '%s': not allowed!" % (name,))
        elif _p is not None and create==True:
            raise ParameterSpaceException("Cannot create parameter '%s': exists!" % (name,))
        elif _p is None and create==True:
            self._all_pars[name] = Parameter(name, value)
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

        self._all_pars[alias] = _p
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


class ParameterSpaceException(Exception):
    pass