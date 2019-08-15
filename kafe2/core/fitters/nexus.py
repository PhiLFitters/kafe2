import abc
import inspect
import six
import sys
import weakref
from ast import parse
from collections import OrderedDict


NODE_VALUE_DEFAULT = 1.0


class NodeException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class NodeBase(object):
    """
    Abstract class. Defines the minimal interface required by all specializations.
    """

    RESERVED_PARAMETER_NAMES = ('__all__', '__real__')

    def __init__(self, name, parent_nexus=None):
        self.name = name
        self.nexus = parent_nexus

        self._stale = False

    @staticmethod
    def check_parameter_name_raise(parameter_name):
        if parameter_name in NodeValue.RESERVED_PARAMETER_NAMES:
            raise NodeException("Invalid parameter name: '%s' is a reserved keyword!"
                                % (parameter_name,))
        try:
            parse('dict(%s=0.123)' % (parameter_name,))
        except SyntaxError:
            raise NodeException("Invalid parameter name '%s'. Must be Python identifier!"
                                % (parameter_name,))

    @property
    def stale(self):
        return self._stale

    @property
    def nexus(self):
        return self._nexus_weak_ref()

    @nexus.setter
    def nexus(self, nexus_ref):
        self._nexus_weak_ref = weakref.ref(nexus_ref)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self.check_parameter_name_raise(name)
        self._name = name

    @abc.abstractproperty
    def value(self):
        pass

    def mark_for_update(self):
        #logger.debug("Marking for update: %s", self)
        self._stale = True

    def notify_dependencies(self):
        #logger.debug("Notifying dependencies: %s", self)
        if self.nexus:
            self.nexus.notify_dependencies(self)

    def __str__(self):
        return "ParameterBase('%s', nexus=%s) [%d]" % (self.name, self.nexus, id(self))


class NodeValue(NodeBase):
    def __init__(self, name, value, parent_nexus=None):
        super(NodeValue, self).__init__(name, parent_nexus=parent_nexus)
        self._value = value
        self._stale = False

        #logger.debug("Created: %s", self)

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
        return "Parameter('%s', value='%s', nexus=%s) [%d]" % (self.name, self.value, self.nexus, id(self))


class NodeFunction(NodeBase):
    """All keyword arguments of the function must be parameters registered in the parameter space."""
    def __init__(self, function_handle, function_name=None, parent_nexus=None):
        _fname = function_name if function_name is not None else function_handle.__name__
        super(NodeFunction, self).__init__(_fname, parent_nexus=parent_nexus)
        self.func = function_handle
        self._value = 0.0
        #self._par_value_cache = dict()
        self._par_value_cache = []
        self._stale = True

        #logger.debug("Created: %s", self)

    def __call__(self):
        #return self._func(**self._par_value_cache)
        return self._func(*self._par_value_cache)

    @property
    def parameter_names(self):
        return self._func_varnames

    @parameter_names.setter
    def parameter_names(self, para_names):
        self._func_varcount = len(para_names)
        self._func_varnames = para_names
        self._stale = True

    def _update(self):
        #logger.debug("Recalculating: %s", self)
        if self.nexus is None:
            return

        #self._par_value_cache = dict()
        self._par_value_cache = []
        _pns = self._func_varnames
        _ps = self.nexus.get_by_name(_pns)
        for _pn, _p in zip(_pns, _ps):
            #logger.debug("Update: %s = %s (%s)", _pn, _p.value, _p)
            #self._par_value_cache[_pn] = _p.value
            self._par_value_cache.append(_p.value)

        self._value = self()
        self._stale = False

    @property
    def value(self):
        #logger.debug("Request function value: %s", self)
        #logger.debug("Stale? %s", self.stale)
        if self.stale:
            #logger.debug("Is stale -> recalculate!")
            self._update()
        #logger.debug("Value: %s", self._value)
        return self._value

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, function_handle):
        self._func = function_handle
        # do introspection
        self._func_varcount = self._func.__code__.co_argcount
        self._func_varnames = inspect.getargspec(self._func)[0]
        self._stale = True

    def __str__(self):
        return "ParameterFunction('%s', function_handle=%s, nexus=%s) [%d]" % (self.name, self._func, self.nexus, id(self))


# ----------------------------------------------

class NexusException(Exception):
    pass

class Nexus(object):
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

        # "GetterCache": speed up parameter access
        self.__getter_cache = dict()

        self.__nexus_stale = True  # need to rebuild nexus?

        self._dependency_graph = dict()
        self._dependency_graph_stale = False

    # -- private methods

    @staticmethod
    def _check_parameter_name_invalid_reserved_raise(parameter_name):
        try:
            parse('dict(%s=0.123)' % (parameter_name,))
        except SyntaxError:
            raise NodeException("Invalid parameter name '%s'. Must be Python identifier!"
                                % (parameter_name,))

    def _check_for_dependency_cycles_raise(self):
        """Check for dependency cycles. Recursive and expensive!"""
        _seen = list()
        def check_cycle_from_vertex(v):
            _seen.add(v)
            for _dep in self._dependency_graph.get(v, tuple()):
                if _dep in _seen or check_cycle_from_vertex(_dep):
                    _m = ' <- '.join(_seen)
                    raise NexusException("Cyclic parameter dependency detected: %s!" % (_m,))
            _seen.remove(v)

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
        for _pn, _p in six.iteritems(self.__map_par_name_to_par_obj):
            if _p not in _first_seen_par_obj_names_ids and _p not in _first_seen_par_function_obj_names_ids:
                if isinstance(_p, NodeFunction):
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

        for _p, (_pn, _pid) in six.iteritems(_first_seen_par_obj_names_ids):
            self.__nexus_list_main_par_objects.append(_p)
            self.__nexus_list_main_par_names.append(_pn)
            self.__nexus_map_all_par_names_to_id[_pn] = _pid
        for _p, (_pn, _pid) in six.iteritems(_first_seen_par_function_obj_names_ids):
            self.__nexus_list_main_par_objects.append(_p)
            self.__nexus_list_main_par_names.append(_pn)
            self.__nexus_map_all_par_names_to_id[_pn] = _pid
        for _p, (_pn, _pid) in six.iteritems(_subsequent_par_obj_names_ids):
            self.__nexus_list_main_par_objects.append(_p)
            self.__nexus_list_main_par_names.append(_pn)
            self.__nexus_map_all_par_names_to_id[_pn] = _pid
        for _p, (_pn, _pid) in six.iteritems(_subsequent_par_function_obj_names_ids):
            self.__nexus_list_main_par_objects.append(_p)
            self.__nexus_list_main_par_names.append(_pn)
            self.__nexus_map_all_par_names_to_id[_pn] = _pid

        self.__nexus_real_dim = _real_dim
        self.__nexus_stale = False
        self.__getter_cache = dict()  # rebuilding nexus invalidates getter cache

    # -- heuristic getters
    #
    # def _get_par_obj(self, par_spec):
    #     if self.__nexus_stale:
    #         self._rebuild_nexus()
    #
    #     # check the cache
    #     _found_par = self.__getter_cache.get(par_spec, None)
    #     if _found_par is not None:
    #         return _found_par
    #
    #     # long lookup and fill cache
    #     if isinstance(par_spec, ParameterBase):
    #         # is param object
    #         if par_spec not in self.__nexus_list_main_par_objects:
    #             raise ParameterSpaceException("%s is not registered in this parameter space!" % (par_spec,))
    #         self.__getter_cache[par_spec] = par_spec
    #         return par_spec
    #     try:
    #         int(par_spec)
    #         # is ID
    #         if par_spec < 0 or par_spec > len(self.__nexus_list_main_par_names):
    #             raise ParameterSpaceException("No parameter with ID '%d'!" % (par_spec,))
    #         _found_par = self.__nexus_list_main_par_objects[par_spec]
    #         self.__getter_cache[par_spec] = _found_par
    #         return _found_par
    #     except (TypeError, ValueError):
    #         # is probably par name (str/unicode)
    #         if par_spec not in self.__nexus_list_main_par_names:
    #             raise ParameterSpaceException("No parameter with name '%s'!" % (par_spec,))
    #         _id = self.__nexus_list_main_par_names.index(par_spec)
    #         _found_par = self.__nexus_list_main_par_objects[_id]
    #         self.__getter_cache[par_spec] = _found_par
    #         return _found_par
    #
    # def _get_par_main_name(self, par_spec):
    #     if self.__nexus_stale:
    #         self._rebuild_nexus()
    #     try:
    #         int(par_spec)
    #         # is ID
    #         if par_spec < 0 or par_spec > len(self.__nexus_list_main_par_names):
    #             raise ParameterSpaceException("No parameter with ID '%d'!" % (par_spec,))
    #         return self.__nexus_list_main_par_names[par_spec]
    #     except (TypeError, ValueError):
    #         if isinstance(par_spec, ParameterBase):
    #             # is param object
    #             if par_spec not in self.__nexus_list_main_par_objects:
    #                 raise ParameterSpaceException("%s is not registered in this parameter space!" % (par_spec,))
    #             _id = self.__nexus_list_main_par_objects.index(par_spec)
    #             return self.__nexus_list_main_par_names[_id]
    #         else:
    #             # is probably par name (str/unicode)
    #             if par_spec not in self.__nexus_list_main_par_names:
    #                 raise ParameterSpaceException("No parameter with name '%s'!" % (par_spec,))
    #             return par_spec
    #
    # def _get_par_id(self, par_spec):
    #     if self.__nexus_stale:
    #         self._rebuild_nexus()
    #     try:
    #         int(par_spec)
    #         if par_spec < 0 or par_spec > len(self.__nexus_list_main_par_names):
    #             raise ParameterSpaceException("No parameter with ID '%d'!" % (par_spec,))
    #         # is ID
    #         return par_spec
    #     except (TypeError, ValueError):
    #         if isinstance(par_spec, ParameterBase):
    #             # is param object
    #             if par_spec not in self.__nexus_list_main_par_objects:
    #                 raise ParameterSpaceException("%s is not registered in this parameter space!" % (par_spec,))
    #             return self.__nexus_list_main_par_objects.index(par_spec)
    #         else:
    #             # is probably par name (str/unicode)
    #             if par_spec not in self.__nexus_list_main_par_names:
    #                 raise ParameterSpaceException("No parameter with name '%s'!" % (par_spec,))
    #             return self.__nexus_list_main_par_names.index(par_spec)


    # -- helper methods

    def _get_one_by_name(self, name, default):
        return self.__map_par_name_to_par_obj.get(name, default)

    def _new_one(self, name, value):
        _p = self.__map_par_name_to_par_obj.get(name)
        if _p is not None:
            raise NexusException("Cannot create parameter '%s': exists!" % (name,))
        else:
            try:
                self.__map_par_name_to_par_obj[name] = NodeValue(name, value, parent_nexus=self)
            except NodeException as pe:
                # re-raise ParameterException as ParameterSpaceException
                raise NexusException(pe)
            self.__nexus_stale = True

    def _set_one(self, name, value):
        _p = self._get_one_by_name(name, None)
        if _p is None:
            raise NexusException("Cannot set parameter '%s': no such parameter!" % (name,))
        else:
            _p.value = value

    def _set_function_one(self, name, function_handle):
        _p = self._get_one_by_name(name, None)
        if _p is None:
            raise NexusException("Cannot set function parameter '%s': no such parameter!" % (name,))
        else:
            _p.func = function_handle

    def _new_alias_one(self, alias, name):
        _p = self.__map_par_name_to_par_obj.get(name)
        if _p is None:
            raise NexusException("Cannot create alias of non-existent parameter '%s'!" % (name,))

        _pa = self.__map_par_name_to_par_obj.get(alias)
        if _pa is not None:
            raise NexusException("Cannot create alias '%s': exists!" % (alias,))

        try:
            NodeBase.check_parameter_name_raise(alias)
            self.__map_par_name_to_par_obj[alias] = _p
        except NodeException as pe:
            # re-raise ParameterException as ParameterSpaceException
            raise NexusException(pe)

        self.__nexus_stale = True

    # def _get_dependent_parameter_objects(self, source_par_obj, default=tuple()):
    #     if self._dependency_graph_stale:
    #         self._check_for_dependency_cycles_raise()
    #         self._dependency_graph_stale = False
    #     return self._dependency_graph.get(source_par_obj, default)
    #
    # def _notify_dependent_parameter_objects(self, source_par_obj):
    #     #logger.debug("Notifying targets for source: %s", source)
    #     print "Notifying targets for source: %s" % (source_par_obj,)
    #     for _target_par_obj in self._get_dependent_parameter_objects(source_par_obj, tuple()):
    #         _target_par_obj.mark_for_update()
    #         #logger.debug("Notifying target: %s", _target_par_obj)
    #         print "Notifying target: %s" % (_target_par_obj,)
    #         self._notify_dependent_parameter_objects(_target_par_obj)

    # -- public interface

    # properties

    @property
    def dimension(self):
        if self.__nexus_stale:
            self._rebuild_nexus()
        return self.__nexus_real_dim

    @property
    def parameter_values_dict(self):
        if self.__nexus_stale:
            self._rebuild_nexus()
        _par_n_id_dict = self.__nexus_map_all_par_names_to_id
        _par_n_par_dict = OrderedDict([(_pn, self._get_one_by_name(_pn, None)) for _pn in _par_n_id_dict])
        _par_n_val_dict = OrderedDict([(_pn, _p.value) for _pn, _p in six.iteritems(_par_n_par_dict) if _p is not None])
        return _par_n_val_dict

    @property
    def parameter_names(self):
        if self.__nexus_stale:
            self._rebuild_nexus()
        return self.__nexus_map_all_par_names_to_id.keys()

    # getters

    def get(self, par_spec=None, default=None):
        """
        Note::
            This is slow and not meant for use by internal interfaces or applications geared towards performance!
            For these, use the dedicated getters #TODO#
        """
        if par_spec is None or par_spec == "__all__":
            # return all parameters
            return [self._get_one_by_name(_n, default) for _n in self.__map_par_name_to_par_obj.keys()]
        elif par_spec == "__real__":
            # return all parameters that are not aliases
            if self.__nexus_stale:
                self._rebuild_nexus()
            return self.__nexus_list_main_par_objects[:self.__nexus_real_dim]
        else:
            return self.get_by_name(par_spec, default)

    def get_values(self, name=None, default=None):
        """
        Note::
            This is slow and not meant for use by internal interfaces or applications geared towards performance!
            For these, use the dedicated getters #TODO#
        """
        _l = self.get(name, default)
        if isinstance(_l, list):
            return [_p.value if _p is not None else None for _p in _l]
        else:
            return _l.value if _l is not None else None

    def get_by_name(self, name, default=None):
        if self.__nexus_stale:
            self._rebuild_nexus()
        try:
            iter(name)
            if isinstance(name, six.string_types[0]):
                # return one parameter
                return self._get_one_by_name(name, default)
            else:
                return [self.get_by_name(_n, default) for _n in name]
        except TypeError:
            return self._get_one_by_name(name, default)

    def get_values_by_name(self, name, default=None):
        _l = self.get_by_name(name, default)
        if isinstance(_l, list):
            return [_p.value if _p is not None else None for _p in _l]
        else:
            return _l.value if _l is not None else None

    # create new parameters

    def new(self, **kwargs):
        for k, v in six.iteritems(kwargs):
            self._new_one(k, v)

    def new_function(self, function_handle, function_name=None, add_unknown_parameters=False, wire_parameters=True):
        _p = self.__map_par_name_to_par_obj.get(function_handle)
        if _p is not None:
            raise NexusException("Cannot create parameter '%s': exists!" % (function_handle,))
        else:
            try:
                _pf = NodeFunction(function_handle, function_name=function_name, parent_nexus=self)
                self.__map_par_name_to_par_obj[_pf.name] = _pf
                #self.add_dependency(target=_fname, sources=_pf.parameter_names)
                for _pf_par_name in _pf.parameter_names:
                    if add_unknown_parameters:
                        _pf_par_obj = self.get_by_name(_pf_par_name)
                        if _pf_par_obj is None:
                            self.new(**{_pf_par_name: NODE_VALUE_DEFAULT})
                    if wire_parameters:
                        self.add_dependency(source=_pf_par_name, target=_pf.name)
            except NodeException as pe:
                # re-raise NodeException as NexusException
                raise NexusException(pe.message)
            self.__nexus_stale = True

    # change parameter values

    def set(self, **kwargs):
        for k, v in six.iteritems(kwargs):
            self._set_one(k, v)

    def set_function(self, **kwargs):
        for k, v in six.iteritems(kwargs):
            self._set_function_one(k, v)

    def set_function_parameter_names(self, function_name, parameter_names):
        _func_node = self.get(par_spec=function_name, default=None)
        _func_node.parameter_names = parameter_names
    # parameter aliases

    def new_alias(self, **kwargs):
        for k, v in six.iteritems(kwargs):
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

    # handle parameter dependencies

    def add_dependency(self, source, target):
        #logger.debug("Adding Dependency: source=%s,  target=%s ", source, target)
        #print "Adding Dependency: source=%s,  target=%s " % (source, target)
        _src_par_obj = self.get_by_name(source)
        if _src_par_obj is None:
            raise NexusException("Cannot create parameter dependency: Source parameter '%s' does not exist!" % (source,))

        if _src_par_obj not in self._dependency_graph:
            self._dependency_graph[_src_par_obj] = []
            self._dependency_graph_stale = True

        _target_par_obj = self.get_by_name(target)

        if _target_par_obj is None:
            raise NexusException("Cannot create parameter dependency: Target parameter '%s' does not exist!" % (target,))

        self._dependency_graph[_src_par_obj].append(_target_par_obj)
        self._dependency_graph_stale = True

    # def get_dependencies_OLD(self, source, default=tuple()):
    #     _src_par_obj = self.get_by_name(source)
    #     return self._get_dependent_parameter_objects(_src_par_obj)
    #
    # def notify_dependencies_OLD(self, source):
    #     _src_par_obj = self.get_by_name(source)
    #     print source, _src_par_obj
    #     return self._notify_dependent_parameter_objects(_src_par_obj)

    def get_dependencies(self, source, default=tuple()):
        if self._dependency_graph_stale:
            self._check_for_dependency_cycles_raise()
            self._dependency_graph_stale = False
        _deps = self._dependency_graph.get(source, None)
        if _deps is None:
            # if nothing found, try again assuming 'source' is a parameter name, not a parameter object...
            _src_obj = self.get_by_name(source)
            _deps = self._dependency_graph.get(_src_obj, default)
            return _deps
        return _deps

    def notify_dependencies(self, source):
        for _target in self.get_dependencies(source, tuple()):
            try:
                _target.mark_for_update()
            except AttributeError:
                _target = self.get_by_name(_target)
                _target.mark_for_update()
            finally:
                self.notify_dependencies(_target)

    def print_state(self, output_stream=sys.stdout):
        for _par_name, _par_obj in six.iteritems(self.__map_par_name_to_par_obj):

            if isinstance(_par_obj, NodeFunction):
                sys.stdout.write("{}({})".format(_par_name, ", ".join(_par_obj.parameter_names)))
            else:
                sys.stdout.write("{}".format(_par_name))

            _content = "{}".format(_par_obj.value)

            # if content repr has newlines, display as indented block
            if '\n' in _content:
                _content = "\n" + _content
                _content = _content.replace("\n", "\n\t")
            sys.stdout.write(" = {}\n".format(_content))