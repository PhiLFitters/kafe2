"""

"""

import abc
import weakref

from collections import OrderedDict
from ast import parse


class ParameterException(Exception):
    pass


class ParameterSpaceException(Exception):
    pass


class ParameterBase(object):
    """
    Abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    RESERVED_PARAMETER_NAMES = ('__all__', '__real__')

    def __init__(self, name, parent_parameter_space=None):
        self.name = name
        self.parameter_space = parent_parameter_space

        self._stale = False

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
    def stale(self):
        return self._stale

    @property
    def parameter_space(self):
        return self._parameter_space_weak_ref()

    @parameter_space.setter
    def parameter_space(self, parameter_space_ref):
        self._parameter_space_weak_ref = weakref.ref(parameter_space_ref)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._check_parameter_name_raise(name)
        self._name = name

    @abc.abstractproperty
    def value(self):
        pass

    def mark_for_update(self):
        print "Marking for update: %s" % (self,)
        self._stale = True

    def notify_dependencies(self):
        print "Notifying dependencies: %s" % (self,)
        if self.parameter_space:
            self.parameter_space.notify_dependencies(self)

    def __str__(self):
        return "ParameterBase('%s', parameter_space=%s) [%d]" % (self.name, self.parameter_space, id(self))

class Parameter(ParameterBase):
    def __init__(self, name, value, parent_parameter_space=None):
        super(Parameter, self).__init__(name, parent_parameter_space=parent_parameter_space)
        self._value = value
        self._stale = False

        print "Created: %s" % (self,)

    @property
    def value(self):
        return self._value

    def mark_for_update(self):
        # don't mark simple parameters for update
        pass

    @value.setter
    def value(self, value):
        self._value = value
        self.notify_dependencies()

    def update(self):
        # TODO: handle simple dependencies?
        # simple parameters are always up-to-date
        return

    def __str__(self):
        return "Parameter('%s', value='%g', parameter_space=%s) [%d]" % (self.name, self.value, self.parameter_space, id(self))


class ParameterFunction(ParameterBase):
    """All keyword arguments of the function must be parameters registered in the parameter space."""
    def __init__(self, function_handle, parent_parameter_space=None):
        super(ParameterFunction, self).__init__(function_handle.__name__, parent_parameter_space=parent_parameter_space)
        self.func = function_handle
        self._value = 0.0
        self._par_value_cache = dict()
        self._stale = True

        print "Created: %s" % (self,)

    def __call__(self):
        return self._func(**self._par_value_cache)

    @property
    def parameter_names(self):
        return self._func_varnames

    def _update(self):
        print "Recalculating: %s" % (self,)
        if self.parameter_space is None:
            return

        self._par_value_cache = dict()
        _pns = self._func_varnames
        _ps = self.parameter_space.get(_pns)
        for _pn, _p in zip(_pns, _ps):
            print " Update: %s = %g (%s)" % (_pn, _p.value, _p)
            self._par_value_cache[_pn] = _p.value

        self._value = self()
        self._stale = False

    @property
    def value(self):
        print "\nRequest function value: %s" % (self,)
        print "Stale?", self.stale
        if self.stale:
            print " Is stale -> recalculate!"
            self._update()
        print " Value:", self._value
        return self._value

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, function_handle):
        self._func = function_handle
        # do introspection
        self._func_varcount = self._func.func_code.co_argcount
        self._func_varnames = self._func.func_code.co_varnames

    def __str__(self):
        return "ParameterFunction('%s', function_handle=%s, parameter_space=%s) [%d]" % (self.name, self._func, self.parameter_space, id(self))


class ParameterSpace(object):
    """Manages `Parameter` aliasing and constructing value vectors/error vectors/error matrices for arbitrary subsets
    of the parameter space."""

    def __init__(self):
        self.__map_par_name_to_par_obj = OrderedDict()  # keys are parameter names, values are `Parameter` objects
        # dict(a1=POa, b1=POb, fa1=PFOa, a2=POa)

        # "Nexus": metadata about parameter aliases
        self.__nexus_list_main_par_objects = []  # ID->Parameter, use reverse lookup for par ID by `Parameter` object
        # list(POa, POb, PFOa)
        self.__nexus_list_main_par_names = []    # ID->'name', use reverse lookup for par ID by name
        # list('a1', 'b1', 'fa1')
        self.__nexus_real_dim = 0  # first N parameter IDs (these are neither aliases nor functions)
        # 2
        self.__nexus_map_all_par_names_to_id = OrderedDict()  # 'name'->ID, use reverse lookup for par ID by name
        # dict(a1=0, b1=1, fa1=2, a2=0)

        self.__nexus_stale = True  # need to rebuild nexus?

        self._dependency_graph = dict()
        self._dependency_graph_stale = False

        self._covariances = None  # this is to be populated by a `Fit` object

    # -- private methods

    @staticmethod
    def _check_parameter_name_invalid_reserved_raise(parameter_name):
        try:
            parse('dict(%s=0.123)' % (parameter_name,))
        except SyntaxError:
            raise ParameterException("Invalid parameter name '%s'. Must be Python identifier!"
                                     % (parameter_name,))

    def _check_for_dependency_cycles_raise(self):
        """Check for dependency cycles. Recursive and expensive!"""
        _seen = list()
        def check_cycle_from_vertex(v):
            _seen.add(v)
            for _dep in self._dependency_graph.get(v, tuple()):
                if _dep in _seen or check_cycle_from_vertex(_dep):
                    _m = ' <- '.join(_seen)
                    raise ParameterSpaceException("Cyclic parameter dependency detected: %s!" % (_m,))
            _seen.remove(v)

    # def _rebuild_nexus(self):
    #     _seen = []
    #     self.__nexus_list_real_par_names = []
    #     self.__nexus_map_par_id_to_par_main_name = {}
    #     _ids = {}
    #     _dim = 0
    #     for _pn, _p in self.__map_par_name_to_par_obj.iteritems():
    #         if _p not in _seen:
    #             _seen.append(_p)
    #             if not isinstance(_p, ParameterFunction):
    #                 self.__nexus_list_real_par_names.append(_p)
    #                 self.__nexus_map_par_id_to_par_main_name[_dim] = _pn
    #             _dim += 1
    #         _ids[_pn] = _seen.index(_p)
    #
    #     self.__nexus_map_par_name_to_par_sequential_id = _ids
    #     self.__nexus_dim = _dim
    #     self.__nexus_stale = False

    def _rebuild_nexus(self):

        # "Nexus": metadata about parameter aliases
        self.__nexus_list_main_par_objects = []  # ID->Parameter, use reverse lookup for par ID by `Parameter` object
        # list(POa, POb, PFOa)
        self.__nexus_list_main_par_names = []    # ID->'name', use reverse lookup for par ID by name
        # list('a1', 'b1', 'fa1')
        self.__nexus_real_dim = 0  # first N parameter IDs (these are neither aliases nor functions)
        # 2
        self.__nexus_map_all_par_names_to_id = OrderedDict()  # 'name'->ID, use reverse lookup for par ID by name
        # dict(a1=0, b1=1, fa1=2, a2=0)

        _first_seen_par_obj_names_ids = OrderedDict()
        _first_seen_par_function_obj_names_ids = OrderedDict()
        _subsequent_par_obj_names_ids = OrderedDict()
        _subsequent_par_function_obj_names_ids = OrderedDict()
        _real_dim = 0
        _main_id = 0
        for _pn, _p in self.__map_par_name_to_par_obj.iteritems():
            if _p not in _first_seen_par_obj_names_ids and _p not in _first_seen_par_function_obj_names_ids:
                if isinstance(_p, ParameterFunction):
                    _first_seen_par_function_obj_names_ids[_p] = (_pn, _main_id)
                else:
                    _first_seen_par_obj_names_ids[_p] = (_pn, _main_id)
                    _real_dim += 1
                _main_id += 1
            elif _p in _first_seen_par_function_obj_names_ids:
                _id = _first_seen_par_function_obj_names_ids[_p][1]
                _subsequent_par_function_obj_names_ids[_p] = (_pn, _id)
            else:
                _id = _first_seen_par_obj_names_ids[_p][1]
                _subsequent_par_obj_names_ids[_p] = (_pn, _id)

        for _p, (_pn, _pid) in _first_seen_par_obj_names_ids.iteritems():
            self.__nexus_list_main_par_objects.append(_p)
            self.__nexus_list_main_par_names.append(_pn)
            self.__nexus_map_all_par_names_to_id[_pn] = _pid
        for _p, (_pn, _pid) in _first_seen_par_function_obj_names_ids.iteritems():
            self.__nexus_list_main_par_objects.append(_p)
            self.__nexus_list_main_par_names.append(_pn)
            self.__nexus_map_all_par_names_to_id[_pn] = _pid
        for _p, (_pn, _pid) in _subsequent_par_obj_names_ids.iteritems():
            self.__nexus_list_main_par_objects.append(_p)
            self.__nexus_list_main_par_names.append(_pn)
            self.__nexus_map_all_par_names_to_id[_pn] = _pid
        for _p, (_pn, _pid) in _subsequent_par_function_obj_names_ids.iteritems():
            self.__nexus_list_main_par_objects.append(_p)
            self.__nexus_list_main_par_names.append(_pn)
            self.__nexus_map_all_par_names_to_id[_pn] = _pid

        self.__nexus_real_dim = _real_dim
        self.__nexus_stale = False

    def _get_par_obj(self, par_spec):
        if self.__nexus_stale:
            self._rebuild_nexus()
        if isinstance(par_spec, ParameterBase):
            # is param object
            if par_spec not in self.__nexus_list_main_par_objects:
                raise ParameterSpaceException("%s is not registered in this parameter space!" % (par_spec,))
            return par_spec
        try:
            int(par_spec)
            # is ID
            if par_spec < 0 or par_spec > len(self.__nexus_list_main_par_names):
                raise ParameterSpaceException("No parameter with ID '%d'!" % (par_spec,))
            return self.__nexus_list_main_par_objects[par_spec]
        except (TypeError, ValueError):
            # is probably par name (str/unicode)
            if par_spec not in self.__nexus_list_main_par_names:
                raise ParameterSpaceException("No parameter with name '%s'!" % (par_spec,))
            _id = self.__nexus_list_main_par_names.index(par_spec)
            return self.__nexus_list_main_par_objects[_id]

    def _get_par_main_name(self, par_spec):
        if self.__nexus_stale:
            self._rebuild_nexus()
        try:
            int(par_spec)
            # is ID
            if par_spec < 0 or par_spec > len(self.__nexus_list_main_par_names):
                raise ParameterSpaceException("No parameter with ID '%d'!" % (par_spec,))
            return self.__nexus_list_main_par_names[par_spec]
        except (TypeError, ValueError):
            if isinstance(par_spec, ParameterBase):
                # is param object
                if par_spec not in self.__nexus_list_main_par_objects:
                    raise ParameterSpaceException("%s is not registered in this parameter space!" % (par_spec,))
                _id = self.__nexus_list_main_par_objects.index(par_spec)
                return self.__nexus_list_main_par_names[_id]
            else:
                # is probably par name (str/unicode)
                if par_spec not in self.__nexus_list_main_par_names:
                    raise ParameterSpaceException("No parameter with name '%s'!" % (par_spec,))
                return par_spec

    def _get_par_main_name_and_aliases(self, par_spec):
        if self.__nexus_stale:
            self._rebuild_nexus()
        try:
            int(par_spec)
            # is ID
            if par_spec < 0 or par_spec > len(self.__nexus_list_main_par_names):
                raise ParameterSpaceException("No parameter with ID '%d'!" % (par_spec,))
            return [_pn for _pn, _pid in self.__nexus_map_all_par_names_to_id if _pid == par_spec]
        except (TypeError, ValueError):
            if isinstance(par_spec, ParameterBase):
                # is param object
                if par_spec not in self.__nexus_list_main_par_objects:
                    raise ParameterSpaceException("%s is not registered in this parameter space!" % (par_spec,))
                _id = self.__nexus_list_main_par_objects.index(par_spec)
                return self.__nexus_list_main_par_names[_id]
                return [_pn for _pn, _pid in self.__nexus_map_all_par_names_to_id if _id == _pid]
            else:
                # is probably par name (str/unicode)
                if par_spec not in self.__nexus_list_main_par_names:
                    raise ParameterSpaceException("No parameter with name '%s'!" % (par_spec,))
                _id = self.__nexus_list_main_par_names.index(par_spec)
                return [_pn for _pn, _pid in self.__nexus_map_all_par_names_to_id if _id == _pid]

    def _get_par_id(self, par_spec):
        if self.__nexus_stale:
            self._rebuild_nexus()
        try:
            int(par_spec)
            if par_spec < 0 or par_spec > len(self.__nexus_list_main_par_names):
                raise ParameterSpaceException("No parameter with ID '%d'!" % (par_spec,))
            # is ID
            return par_spec
        except (TypeError, ValueError):
            if isinstance(par_spec, ParameterBase):
                # is param object
                if par_spec not in self.__nexus_list_main_par_objects:
                    raise ParameterSpaceException("%s is not registered in this parameter space!" % (par_spec,))
                return self.__nexus_list_main_par_objects.index(par_spec)
            else:
                # is probably par name (str/unicode)
                if par_spec not in self.__nexus_list_main_par_names:
                    raise ParameterSpaceException("No parameter with name '%s'!" % (par_spec,))
                return self.__nexus_list_main_par_names.index(par_spec)

    # # -- private "properties"
    #
    # @property
    # def _name_id_map(self):
    #     if self.__nexus_stale:
    #         self._rebuild_nexus()
    #     return self.__nexus_map_par_name_to_par_sequential_id
    #
    # @property
    # def _real_pars(self):
    #     if self.__nexus_stale:
    #         self._rebuild_nexus()
    #     return self.__nexus_list_real_par_names
    #
    # @property
    # def _dim(self):
    #     if self.__nexus_stale:
    #         self._rebuild_nexus()
    #     return self.__nexus_dim

    # -- public interface

    # def parameter_main_name(self, parameter):
    #     if isinstance(parameter, ParameterBase):
    #         _id = self.__map_par_name_to_par_obj.index(parameter)
    #     else:
    #         _id = self._name_id_map.get(parameter)
    #     return self.__nexus_map_par_id_to_par_main_name[_id]

    def _get_one(self, name, default):
        return self.__map_par_name_to_par_obj.get(name, default)

    def get(self, name=None, default=None):
        if name is None or name == "__all__":
            # return all parameters
            return [self._get_one(_n, default) for _n in self.__map_par_name_to_par_obj.keys()]
        elif name == "__real__":
            # return all parameters that are not aliases
            if self.__nexus_stale:
                self._rebuild_nexus()
            return self.__nexus_list_main_par_objects[:self.__nexus_real_dim]
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

    def _new_one(self, name, value):
        _p = self.__map_par_name_to_par_obj.get(name)
        if _p is not None:
            raise ParameterSpaceException("Cannot create parameter '%s': exists!" % (name,))
        else:
            try:
                self.__map_par_name_to_par_obj[name] = Parameter(name, value, parent_parameter_space=self)
            except ParameterException as pe:
                # re-raise ParameterException as ParameterSpaceException
                raise ParameterSpaceException(pe.message)
            self.__nexus_stale = True

    def new(self, **kwargs):
        for k, v in kwargs.iteritems():
            self._new_one(k, v)

    def new_function(self, function_handle):
        _p = self.__map_par_name_to_par_obj.get(function_handle)
        if _p is not None:
            raise ParameterSpaceException("Cannot create parameter '%s': exists!" % (function_handle,))
        else:
            try:
                _fname = function_handle.__name__
                _pf = self.__map_par_name_to_par_obj[_fname] = ParameterFunction(function_handle, parent_parameter_space=self)
                #self.add_dependency(target=_fname, sources=_pf.parameter_names)
                for _pf_par_name in _pf.parameter_names:
                    self.add_dependency(source=_pf_par_name, target=_fname)
            except ParameterException as pe:
                # re-raise ParameterException as ParameterSpaceException
                raise ParameterSpaceException(pe.message)
            self.__nexus_stale = True

    def _set_one(self, name, value):
        _p = self.__map_par_name_to_par_obj.get(name)
        if _p is None:
            raise ParameterSpaceException("Cannot set parameter '%s': no such parameter!" % (name,))
        else:
            _p.value = value

    def set(self, **kwargs):
        for k, v in kwargs.iteritems():
            self._set_one(k, v)

    def _new_alias_one(self, alias, name):
        _p = self.__map_par_name_to_par_obj.get(name)
        if _p is None:
            raise ParameterSpaceException("Cannot create alias of non-existent parameter '%s'!" % (name,))

        _pa = self.__map_par_name_to_par_obj.get(alias)
        if _pa is not None:
            raise ParameterSpaceException("Cannot create alias '%s': exists!" % (alias,))

        try:
            Parameter._check_parameter_name_raise(alias)
            self.__map_par_name_to_par_obj[alias] = _p
        except ParameterException as pe:
            # re-raise ParameterException as ParameterSpaceException
            raise ParameterSpaceException(pe.message)

        self.__nexus_stale = True

    def new_alias(self, **kwargs):
        for k, v in kwargs.iteritems():
            self._new_alias_one(k, v)

    # def add_dependency(self, target, sources):
        # print "Adding Dependency: target=%s, sources=%r" % (target, sources)
        # if self.get(target) is None:
        #     raise ParameterSpaceException("Cannot create parameter dependency: Target parameter '%s' does not exist!" % (target,))
        #
        # if target not in self._dependency_graph:
        #     self._dependency_graph[target] = []
        #
        # _src_pars = self.get(sources)
        #
        # if isinstance(_src_pars, ParameterBase):
        #     _src_pars = [_src_pars]
        #
        # for _src_par, _src_par_name in zip(_src_pars, sources):
        #     if _src_par is None:
        #         raise ParameterSpaceException("Cannot create parameter dependency: Source parameter '%s' does not exist!" % (_src_par_name,))
        #     self._dependency_graph[target].append(_src_par_name)

    def add_dependency(self, source, target):
        print "Adding Dependency: source=%s,  target=%s " % (source, target)
        _src_par_obj = self._get_par_obj(source)
        if _src_par_obj is None:
            raise ParameterSpaceException("Cannot create parameter dependency: Source parameter '%s' does not exist!" % (source,))

        if _src_par_obj not in self._dependency_graph:
            self._dependency_graph[_src_par_obj] = []
            self._dependency_graph_stale = True

        _target_par_obj = self._get_par_obj(target)

        if _target_par_obj is None:
            raise ParameterSpaceException("Cannot create parameter dependency: Target parameter '%s' does not exist!" % (target,))

        self._dependency_graph[_src_par_obj].append(_target_par_obj)
        self._dependency_graph_stale = True

    def get_dependencies(self, source, default=tuple()):
        if self._dependency_graph_stale:
            self._check_for_dependency_cycles_raise()
        _src_par_obj = self._get_par_obj(source)
        return self._dependency_graph.get(_src_par_obj, default)

    def notify_dependencies(self, source):
        _src_name = self._get_par_main_name(source)
        print "Notifying targets for source: %s" % (source,)
        print " Targets: %r" % (self.get_dependencies(source, tuple()),)
        for _target_par_obj in self.get_dependencies(source, tuple()):
            _target_par_obj.mark_for_update()
            print " Notifying target: %s" % (_target_par_obj,)
            self.notify_dependencies(_target_par_obj)

    @property
    def dimension(self):
        if self.__nexus_stale:
            self._rebuild_nexus()
        return self.__nexus_real_dim


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