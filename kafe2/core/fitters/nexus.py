import abc
import inspect
import numpy as np
import operator
import six
import sys
import uuid
import warnings
import weakref

from ast import parse
from collections import OrderedDict

if six.PY2:
    from funcsigs import signature, Parameter as SigParameter
else:
    from inspect import signature, Parameter as SigParameter


__all__ = [
    "Parameter",
    "Function",
    "Fallback",
    "Tuple",
    "Nexus",
]

# -- helpers

_OPERATORS = {
    # arithmetic
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'truediv': operator.truediv,
    'floordiv': operator.floordiv,
    'mod': operator.mod,
}

# 'div' operator only available in Python 2
if six.PY2:
    _OPERATORS['div'] = operator.div

_OPERATORS = {
    # arithmetic
    'add': (operator.add, True),
    'sub': (operator.sub, True),
    'mul': (operator.mul, True),
    'truediv': (operator.truediv, True),
    'floordiv': (operator.floordiv, True),
    'mod': (operator.mod, True),
    # binary arithmetic
    'lshift': (operator.lshift, True),
    'rshift': (operator.rshift, True),
    # logical
    'and': (operator.and_, True),
    'or': (operator.or_, True),
    'xor': (operator.xor, True),
    # comparison
    #'eq': (operator.eq, False),
    #'ne': (operator.ne, False),
    #'lt': (operator.lt, False),
    #'le': (operator.le, False),
    #'gt': (operator.gt, False),
    #'ge': (operator.ge, False),
    # other
    #'getitem': (operator.getitem, False),
    #'contains': (operator.contains, False),
}

_UNARY_OPERATORS = {
    'invert': operator.invert,
    'neg': operator.neg,
    'pos': operator.pos,
    'len': len,
}

# 'div' operator only available in Python 2
if six.PY2:
    _OPERATORS['div'] = (operator.div, True)

# '_abs' operator only available in Python 3
if six.PY3:
    _UNARY_OPERATORS['abs'] = operator._abs


def _add_common_operators(cls):
    """class decorator for implementing common operator methods for nodes"""

    for name, (op, add_reversed) in list(_OPERATORS.items()):
        setattr(cls, '__{}__'.format(name), cls._make_binop_method(op))
        if add_reversed:
            setattr(cls, '__r{}__'.format(name), cls._make_rbinop_method(op))
    for name, op in _UNARY_OPERATORS.items():
        setattr(cls, '__{}__'.format(name), cls._make_unaryop_method(op))

    return cls


# -- Nodes -------------------------------------

class NodeException(Exception):
    pass


class FallbackError(NodeException):
    pass

@six.add_metaclass(abc.ABCMeta)
class NodeBase(object):
    """
    Abstract class for a computation graph node.
    Defines the minimal interface required by all specializations.
    """

    RESERVED_PARAMETER_NAMES = ('__all__', '__real__', '__root__', '__error__')

    def __init__(self, name=None):
        self.name = name or 'Node_' + uuid.uuid4().hex[:10]

        self._stale = True
        self._frozen = False
        self._callbacks = []
        self._children = []
        self._parents = set()

    def __iter__(self):
        raise TypeError("'{}' is not iterable".format(self.__class__.__name__))

    def __str__(self):
        return "{}('{}')  [{}]".format(
            self.__class__.__name__,
            self.name, hex(id(self))
        )

    def _pprint(self):
        '''return pretty-printed version of node'''
        return "{}('{}')".format(
            self.__class__.__name__,
            self.name
        )

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

    @staticmethod
    def _check_name_raise(name):
        if name in Parameter.RESERVED_PARAMETER_NAMES:
            raise NodeException("Invalid node name: '%s' is a reserved keyword!"
                                % (name,))
        try:
            parse('dict(%s=0.123)' % (name,))
        except SyntaxError:
            raise NodeException("Invalid node name '%s'. Must be Python identifier!"
                                % (name,))

    def _execute_callbacks(self):
        # execute any callback functions
        for _cb in self._callbacks:
            _cb['func'](
                *(_cb['args'] or []),
                **(_cb['kwargs'] or {})
            )

    # -- properties

    @property
    def stale(self):
        return self._stale

    @property
    def frozen(self):
        return self._frozen

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        NodeBase._check_name_raise(name)
        self._name = name

    # -- public API

    def add_child(self, node):
        if not isinstance(node, NodeBase):
            # wrap non-node values inside `Parameter`
            node = Parameter(node)

        self._children.append(node)
        node.add_parent(self)

        return self

    def add_parent(self, node):
        if not isinstance(node, NodeBase):
            raise TypeError(
                "Cannot add parent: expected "
                "node type, got {}".format(type(node))
            )
        self._parents.add(weakref.ref(node))

        return self

    def get_children(self):
        return list(self.iter_children())

    def get_parents(self):
        return list(self.iter_parents())

    def remove_child(self, node):
        if not isinstance(node, NodeBase):
            raise TypeError(
                "Cannot remove child: expected "
                "node type, got {}".format(type(node))
            )
        # remove all instances of the child node from root children
        self._children = [
            _child
            for _child in self._children
            if _child is not node
        ]

        # remove self node from node parents
        node.remove_parent(self)

        return self

    def remove_parent(self, node):
        if not isinstance(node, NodeBase):
            raise TypeError(
                "Cannot remove parent: expected "
                "node type, got {}".format(type(node))
            )

        # remove node from node parents
        self._parents.remove(weakref.ref(node))

        # do *not* remove self node from node children

        return self

    def iter_children(self):
        for _child in self._children:
            yield _child

    def iter_parents(self):
        _out_of_scope_refs = []
        for _p_ref in self._parents:
            _p = _p_ref()  # dereference weakref

            # parent went out of scope
            if _p is None:
                _out_of_scope_refs.append(_p_ref)
                continue

            yield _p

        # cleanup
        if _out_of_scope_refs:
            warnings.warn(
                "{} parent refs went out of scope since last "
                "call to 'iter_parents' and will be removed.".format(
                    len(_out_of_scope_refs)
                ),
                UserWarning
            )
        for _ref in _out_of_scope_refs:
            self._parents.remove(_ref)

    def set_children(self, children):
        _new_children = []
        for _child in children:
            # wrap literal values inside 'Parameter'
            if not isinstance(_child, NodeBase):
                _child = Parameter(_child)

            _new_children.append(_child)

        self._children = _new_children

        for _child in self._children:
            _child.add_parent(self)

        self.mark_for_update()

        return self

    def set_parents(self, parents):
        _new_parents = set()
        for _parent in parents:
            if not isinstance(_parent, NodeBase):
                raise TypeError(
                    "Cannot set parent: expected "
                    "node type, got {}".format(type(node))
                )
            _new_parents.add(weakref.ref(_parent))

        self._parents = _new_parents

        return self

    def notify_parents(self):
        # frozen nodes do not notify their parents
        if self.frozen:
            return

        # execute any callback functions
        self._execute_callbacks()

        for _p in self.iter_parents():
            _p.mark_for_update()
            _p.notify_parents()

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def mark_for_update(self):
        # invalidate node value
        self._stale = True

    def update(self):
        self._stale = False

    def replace(self, other, transfer_children=False):
        '''replace all instances of this node with `other`'''
        if not isinstance(other, NodeBase):
            other = Parameter(other)

        if transfer_children:
            _other_orig_children = other.get_children()
            other.set_children(self.get_children())

        for _p in self.iter_parents():
            other.add_parent(_p)

        # set 'other' as parent of all self's children
        for _child in self.iter_children():
            _child._parents = set([_p if _p is not self else weakref.ref(other) for _p in _child._parents])

        # set 'other' as child of all self's parents
        for _parent in self.iter_parents():
            _parent._children = [_c if _c is not self else other for _c in _parent._children]

        if transfer_children:
            self.set_children(_other_orig_children)

    def register_callback(self, func, args=None, kwargs=None):
        self._callbacks.append(dict(func=func, args=args, kwargs=kwargs))

    def print_descendants(self):
        NodeChildrenPrinter(self).run()


class RootNode(NodeBase):
    """
    A Node that acts as the root node of a graph. Cannot have any parent nodes.
    """

    def __init__(self):
        self._name = '__root__'
        self._stale = True
        self._callbacks = []
        self._children = []
        self._parents = set()  # root node has no parents

    def add_child(self, node):
        NodeBase.add_child(self, node)

        # keep root node children sorted by name
        self._children = sorted(self._children, key=lambda x: x.name)

    def replace(self, other):
        '''replace all instances of this node with `other`'''
        raise TypeError("Root node cannot be replaced!")

    def add_parent(self, parent):
        raise TypeError("Cannot add parent to root node!")

    def iter_parents(self):
        # empty generator
        return iter(())

    def notify_parents(self):
        pass


@_add_common_operators
class ValueNode(NodeBase):
    """
    Partly abstract class for a graph node that can have a value.
    """

    def __init__(self, value=None, name=None):
        NodeBase.__init__(self, name=name)
        self._value = value
        self._stale = False

    def __str__(self):
        return "{}('{}') = {}  [{}]".format(
            self.__class__.__name__,
            self.name, self.value, hex(id(self))
        )

    def _pprint(self):
        '''return pretty-printed version of node'''
        try:
            _val = str(self.value)
        except Exception as e:
            _val = repr(e)
        if '\n' in _val:
            _val = '\n' + _val
        else:
            _val = ' ' + _val
        return NodeBase._pprint(self) + ' =' + _val

    @classmethod
    def _make_binop_method(cls, op):
        def _op_method(self, other):
            if not isinstance(other, NodeBase):
                other = Parameter(other)
            _auto_name = "{}__{}__{}".format(self.name, op.__name__, other.name)

            return Function(
                func=op,
                name=_auto_name,
                parameters=(self, other),
            )

        return _op_method

    @classmethod
    def _make_rbinop_method(cls, op):
        def _op_method(self, other):
            if not isinstance(other, NodeBase):
                other = Parameter(other)
            _auto_name = "{}__r{}__{}".format(self.name, op.__name__, other.name)

            return Function(
                func=op,
                name=_auto_name,
                parameters=(other, self),
            )

        return _op_method

    @classmethod
    def _make_unaryop_method(cls, op):
        def _op_method(self):
            _auto_name = "{}__{}".format(op.__name__, self.name)

            return Function(
                func=op,
                name=_auto_name,
                parameters=(self,),
            )

        return _op_method

    @property
    def value(self):
        '''Retrieve node value.'''
        if self.stale and not self.frozen:
            self.update()

        return self._value

    @value.setter
    def value(self, value):
        '''Set node value.'''
        self._value = value
        # notify parents that a child value has changed
        self.notify_parents()


class Empty(ValueNode):
    """A graph node that acts as a placeholder for a ValueNode.
    Should be replaced by a ValueNode in the finished graph.
    Raises an error if an attempt is made to set or retrieve the value."""
    def __init__(self, name=None):
        NodeBase.__init__(self, name=name)

    def __str__(self):
        return "{}('{}')  [{}]".format(
            self.__class__.__name__,
            self.name, hex(id(self))
        )

    def _pprint(self):
        '''return pretty-printed version of node'''
        return NodeBase._pprint(self)

    @property
    def value(self):
        raise ValueError(
            "Empty node '{}' does not have a value.".format(
                self.name
            )
        )

    @value.setter
    def value(self, value):
        raise ValueError(
            "Cannot set value of empty node '{}'.".format(
                self.name
            )
        )


class Parameter(ValueNode):
    def __init__(self, value, name=None):
        ValueNode.__init__(self, value=value, name=name)
        self._stale = False

    def update(self):
        # simple parameters are always up-to-date
        return


class Alias(ValueNode):
    """
    A Node that only passes on the value
    of another node when evaluated.
    """
    def __init__(self, ref, name=None):
        ValueNode.__init__(self, value=None, name=name)

        self.ref = ref

        self._stale = True

    def _pprint(self):
        '''return pretty-printed version of node'''
        return '{} -> {}'.format(
            NodeBase._pprint(self),
            NodeBase._pprint(self.ref)
        )

    @ValueNode.value.setter
    def value(self, value):
        raise ValueError(
            "Cannot set value of alias node '{}'.".format(
                self.name
            )
        )

    @property
    def ref(self):
        """The node for which this is an alias."""
        return self._children[0]

    @ref.setter
    def ref(self, ref):
        self.set_children([ref])
        ref.add_parent(self)

    def update(self):
        self._value = self.ref.value


class Function(ValueNode):
    """All keyword arguments of the function must be parameters registered in the parameter space."""

    def __init__(self, func, name=None, parameters=None):
        _fname = name or func.__name__
        super(Function, self).__init__(value=None, name=_fname)

        self.func = func
        self.parameters = parameters or []

        self._par_cache = []

    def __call__(self):
        return self._func(*self._par_cache)

    @property
    def parameters(self):
        """The nodes representing the function parameters."""
        return self._children

    @parameters.setter
    def parameters(self, parameters):
        self.set_children(parameters)

    @property
    def func(self):
        """The function handle to be called."""
        return self._func

    @func.setter
    def func(self, function_handle):
        self._func = function_handle
        self._stale = True

    @ValueNode.value.setter
    def value(self, value):
        '''Set node value.'''
        raise NodeException("Function node value cannot be set!")

    def update(self):
        self._par_cache = [
            _par.value
            for _par in self._children
        ]
        self._value = self()
        self._stale = False


class Fallback(ValueNode):

    def __init__(self, try_nodes, exception_type=Exception, name=None):
        super(Fallback, self).__init__(value=None, name=name)
        self.try_nodes = try_nodes
        self._exception_type = exception_type
        self._value = None

        if name is None:
            self.name = '__onfail__'.join([_n.name for _n in self.try_nodes])

    def __call__(self):
        for _node in self.try_nodes:
            try:
                return _node.value
            except self._exception_type:
                pass  # function failed; try next

        raise FallbackError(
            "Error evaluating fallback node '{}': "
            "no alternative succeeded".format(self.name)
        )

    @property
    def try_nodes(self):
        """The nodes representing alternative values."""
        return self._children

    @try_nodes.setter
    def try_nodes(self, try_nodes):
        self.set_children(try_nodes)

    def update(self):
        self._value = self()
        self._stale = False


class Tuple(ValueNode):

    def __init__(self, nodes, name=None):
        super(Tuple, self).__init__(value=None, name=name)
        self.nodes = nodes
        self._value = None

        if name is None and len(self.nodes):
            self.name =  '__'.join([_n.name for _n in self.nodes])

    def __iter__(self):
        for _n in self.nodes:
            yield _n

    def __len__(self):
        return len(self._children)

    def __getitem__(self, index):
        return self.nodes[index]

    def __setitem__(self, index):
        raise NotImplementedError

    @property
    def nodes(self):
        """The nodes representing the tuple content."""
        return self._children

    @nodes.setter
    def nodes(self, nodes):
        self.set_children(nodes)

    def _update(self):
        self._value = tuple([
            _node.value for _node in self.nodes
        ])
        self._stale = False

    @property
    def value(self):
        if self.stale:
            self._update()
        return self._value

    def iter_values(self):
        for _val in self.value:
            yield _val


class Array(Tuple):

    def __init__(self, nodes, name=None, dtype=None):
        super(Tuple, self).__init__(value=None, name=name)
        self._dtype = dtype
        self.nodes = nodes

        if name is None and len(self.nodes):
            self.name =  'array_' + '__'.join([_n.name for _n in self.nodes])

    @Tuple.nodes.setter
    def nodes(self, nodes):
        self.set_children(nodes)
        self._value = np.empty(len(nodes), dtype=self._dtype)

    def _update(self):
        for _i, _node in enumerate(self.nodes):
            self._value[_i] = _node.value

        self._stale = False

    def iter_values(self):
        return self.value.__iter__()


# -- Node visitors

class NodeChildrenPrinter(object):
    def __init__(self, root_node):
        self._root = root_node

    def visit(self, node, indent=''):
        _s = node._pprint()
        if '\n' in _s:
            _s = _s.replace('\n', '\n'+indent+'    ')
        print(indent + '+ ' + _s)

    def run(self, node=None, indent=''):
        node = node if node is not None else self._root

        # visit current node
        self.visit(node, indent=indent)

        # recursively visit children
        for _c in node.iter_children():
            self.run(_c, indent=indent + '  ')


class NodeSubgraphGraphvizSourceProducer(object):
    _NODE_STYLE = {
        Function: dict(
            shape='"box"'),
        Alias: dict(
            shape='"cds"'),
        Fallback: dict(
            shape='"diamond"'),
        Empty: dict(
            shape='"Mcircle"'),
    }
    def __init__(self, root_node, exclude=None):
        self._root = root_node
        self._exclude = exclude or set()
        self._nodes = set()
        self._leaf_nodes = set()
        self._connected_nodes = set()
        self._full_edges = set()
        self._dashed_edges = set()

    def visit(self, node):
        self._nodes.add(node)
        for _c in node.iter_children():
            self._nodes.add(node)
            if node in self._exclude or _c in self._exclude:
                continue
            self._full_edges.add(
                (node, _c)
            )
            self._nodes.add(_c)

        for _p in node.iter_parents():
            self._nodes.add(node)
            if node in self._exclude or _p in self._exclude:
                continue
            self._dashed_edges.add(
                (_p, node)
            )
            self._nodes.add(_p)

    def run(self, node=None, node_chain=None, out_stream=None):
        node = node if node is not None else self._root

        _first_call = node_chain is None
        node_chain = node_chain or set()
        # don't revisit previous nodes
        if node in node_chain:
            return

        # visit current node
        self.visit(node)

        node_chain.add(node)

        # recursively visit children
        for _c in node.iter_children():
            NodeSubgraphGraphvizSourceProducer.run(self, _c, node_chain=node_chain)
        # recursively visit parents
        for _p in node.iter_parents():
            NodeSubgraphGraphvizSourceProducer.run(self, _p, node_chain=node_chain)

        if _first_call:

            out_stream = out_stream if out_stream is not None else sys.stdout

            import functools
            _all_edges = self._full_edges.union(self._dashed_edges)
            _connected_nodes = set(functools.reduce(
                lambda x, y: x+y,
                _all_edges,
            ))
            _isolated_nodes = self._nodes - _connected_nodes
            self._exclude |= _isolated_nodes

            _leaf_nodes = (
                set([n for _, n in _all_edges])
                -
                set([n for n, _ in _all_edges])
            )

            out_stream.write("digraph {} {{\n\n".format(node.name))
            for _n in _connected_nodes:
                if _n in self._exclude:
                    continue
                out_stream.write(
                    '  {0.name} [label=<<i>{0.__class__.__name__}</i>'
                    '<br />{0.name}>, {1}]\n'.format(
                        _n,
                        ', '.join([
                            "{}={}".format(_k, _v)
                            for _k, _v
                            in self._NODE_STYLE.get(_n.__class__, {}).items()
                        ] + (['style=filled', 'fillcolor=yellow'] if _n in _leaf_nodes else [])),
                    )
                )

            out_stream.write('\n')
            for _e in self._full_edges:
                if _e[0] in self._exclude or _e[1] in self._exclude:
                    continue
                out_stream.write("  {0.name} -> {1.name}\n".format(*_e))

            out_stream.write('\n')
            for _e in self._dashed_edges - self._full_edges:
                if _e[0] in self._exclude or _e[1] in self._exclude:
                    continue
                out_stream.write("  {0.name} -> {1.name} [style=dashed]\n".format(*_e))
            out_stream.write("\n}\n")
            out_stream.flush()

            self._full_edges = set()
            self._dashed_edges = set()


class NodeSubgraphGraphvizViewer(NodeSubgraphGraphvizSourceProducer):

    def run(self, node=None, node_chain=None):
        from graphviz import Source

        import tempfile

        out = six.StringIO()
        NodeSubgraphGraphvizSourceProducer.run(
            self, node, node_chain, out_stream=out)

        out.seek(0)
        Source(out.read()).render(tempfile.mktemp('.gv'), view=True)


class NodeCycleChecker(object):
    def __init__(self, root_node):
        self._root = root_node

    def visit(self, node, seen):
        if node in seen:
            _cycle = seen[seen.index(node):] + (node,)
            raise ValueError("Dependent node cycle detected ({})".format(
                ' -> '.join([_n.name for _n in _cycle])
            ))

    def run(self, node=None, seen=None):
        node = node if node is not None else self._root
        seen = seen if seen is not None else tuple()

        # visit current node
        self.visit(node, seen)

        # keep track of nodes encountered
        seen += (node,)

        # recursively visit children
        for _c in node.iter_children():
            self.run(node=_c, seen=seen)


# -- Nexus

class NexusError(Exception):
    pass


class Nexus(object):
    """Object representing an entire computation graph."""

    _VALID_GET_DICT_ERROR_BEHAVIORS = (
        'fail', 'none', 'exception_as_value', 'ignore', 'list')
    _VALID_ADD_EXISTING_BEHAVIORS = ('fail', 'ignore', 'replace', 'replace_if_alias', 'replace_if_empty')

    def __init__(self):
        self._nodes = {
            '__root__': RootNode()
        }
        self._root_ref = weakref.ref(self._nodes['__root__'])  # convenience

    def add(self, node, node_namespace=None, child_namespaces=None, existing_behavior='fail'):
        """Add a node to the nexus.

        The `node` type should be derived from
        :py:class:`~kafe.core.fitters.nexus_v2.NodeBase`. If not, it
        will be wrapped inside a
        :py:class:`~kafe.core.fitters.nexus_v2.Parameter` node.

        Nexus node names must be unique. If a node with the same name
        has already been added, the behavior of this function can
        be configured via `existing_behavior`, which takes one of
        three values:
        - ``fail``: raise a `NexusError`
        - ``replace``: replace the existing node with the one being added
        - ``replace_if_alias``: replace the existing node with the one
            only if the existing node is an alias
        - ``replace_if_empty``: replace the existing node with the one
            only if the existing node is empty
        - ``ignore``: do not add the node

        :param node: node to add
        :type node: :py:class:`~kafe.core.fitters.nexus_v2.NodeBase` or other
        :param node_namespace: namespace in which to add the node
        :type node_namespace: str or None
        :param child_namespaces: namespaces in which to add the node's children if they aren't already part of the Nexus
        :type child_namespaces: iterable of str or None
        :param existing_behavior: how to behave if a node with the same name
            has already been added to the nexus. One of: ``fail`` (default),
            ``replace``, ``replace_if_alias``, ``replace_if_empty``,
            ``ignore``.
        :type existing_behavior: str

        :returns: the added node
        """
        if node_namespace is None:
            _effective_node_name = node.name
        else:
            _effective_node_name = node_namespace + node.name
        if child_namespaces is None:
            child_namespaces = (None,) * len(node.get_children())
        elif isinstance(child_namespaces, str):
            child_namespaces = (child_namespaces,) * len(node.get_children())

        # validate `existing_behavior`
        if existing_behavior not in self._VALID_ADD_EXISTING_BEHAVIORS:
            raise ValueError(
                "Unknown value {!r} for `existing_behavior`: "
                "expected one of {!r}".format(
                    existing_behavior,
                    self._VALID_ADD_EXISTING_BEHAVIORS
                )
            )

        # check if node has already been added
        if _effective_node_name in self._nodes:
            # resolve behavior if node is empty
            if existing_behavior == 'replace_if_empty':
                existing_behavior = (
                    'replace'
                    if isinstance(self._nodes[_effective_node_name], Empty)
                    else 'fail'
                )
            elif existing_behavior == 'replace_if_alias':
                existing_behavior = (
                    'replace'
                    if isinstance(self._nodes[_effective_node_name], Alias)
                    else 'fail'
                )

            # execute behavior
            if existing_behavior == 'fail':
                raise NexusError(
                    "Node '{}' already exists.".format(
                        _effective_node_name
                    )
                )
            elif existing_behavior == 'replace':
                # add all dependent children to the nexus first
                for _child, _child_namespace in zip(node.iter_children(), child_namespaces):
                    self.add(_child, node_namespace=_child_namespace, child_namespaces=_child_namespace,
                             existing_behavior='ignore')
                # replace node
                self._nodes[_effective_node_name].replace(node)
            elif existing_behavior == 'ignore':
                return  #
            else:
                assert False  # should not get here

        # node has not been added before
        else:
            # add all dependent children to the nexus first
            for _child, _child_namespace in zip(node.iter_children(), child_namespaces):
                self.add(_child, node_namespace=_child_namespace, child_namespaces=_child_namespace,
                         existing_behavior='ignore')

            # add new node as child of root node
            self._root_ref().add_child(node)

            # remove nexus root from all children's parents (if present)
            for _child in node.iter_children():
                try:
                    self._root_ref().remove_child(_child)
                except KeyError:
                    pass

        # (re)map name to point to node
        self._nodes[_effective_node_name] = node

        # check for cycles
        NodeCycleChecker(self._root_ref()).run()

        return node

    def add_function(self, func, func_name=None, par_names=None, function_namespace=None, parameter_namespaces=None,
                     existing_behavior='fail'):
        """Construct a `Function` node from a Python function and
        add it to the nexus.

        The object passed as `func` should be a native Python function.
        If other callables (e.g. `lambda` expressions) are passed, they
        must either have a `__name__` parameter or `func_name` must be provided.
        The function node name is set to `func_name` (if provided) or
        `func.__name__`.

        For every parameter of the function, there should already be a
        nexus node with the same name as the parameter name. If they
        exist, they will be registered automatically as dependencies of
        the function node, meaning that the function value will be
        recalculated every time a parameter changes value.

        In case the function parameter names do not correspond exactly
        to the names of the parameter nodes in the nexus, a list of
        node names to use as parameters can be specified via the
        `par_names` keyword,

        If any of the specified parameter nodes does not exist, it will
        be created. In case a default value for the parameter is
        specified in the function signature, a
        :py:class:`~kafe.core.fitters.nexus.Parameter` node will be
        created and initialized to that value. If no default is
        specified, an :py:class:`~kafe.core.fitters.nexus.Empty` node
        will be created. It should be replaced by a non-empty node
        before the function is evaluated.

        Any variable-length arguments (``*args``) and keyword arguments
        (``**kwargs``) in the function signature are ignored when
        creating parameters.

        If a node with the same name as the function node
        has already been added, the behavior of this function can
        be configured via `existing_behavior`. For details, see
        :py:meth:`~kafe.core.fitters.nexus.Nexus.add`.

        :param func: function to add as node
        :type func: function
        :param func_name: node name to use instead of function name
        :type func_name: str
        :param par_names: node name to use as function parameters
        :type par_names: list of str
        :param function_namespace: namespace in which to add the function
        :type function_namespace: str or None
        :param parameter_namespaces: namespaces in which to add the function parameters
        :type parameter_namespaces: iterable of str or None
        :param existing_behavior: how to behave if a node with the same
            name has already been added to the nexus. One of: ``fail``
            (default), ``replace``, ``replace_if_empty``, ``ignore``.
        :type existing_behavior: str

        :returns: the added function node
        """
        # resolve function name
        func_name = func_name or func.__name__

        # resolve parameter names
        _sig = signature(func)

        if par_names is None:
            # take child node names from signature
            par_names = [_p.name for _p in _sig.parameters.values()]

        # assert compatibility between function signature
        # and supplied node names
        try:
            _ba = _sig.bind(*par_names)
        except TypeError:
            raise ValueError(
                "Error adding function: supplied "
                "parameter nodes ({}) are not compatible with "
                "function signature {!r}!".format(
                    ', '.join(map(repr, par_names)),
                    _sig
                )
            )

        _args = _ba.arguments.items()
        _args_defaults = [_a.default for _a in _sig.parameters.values()]

        assert len(_args_defaults) >= len(_args)  # barring any unbound *args, **kwargs

        # resolve parameter nodes
        _pars = []
        for (_arg_name, _node_spec), _arg_default in zip(_args, _args_defaults):

            # fail on **kwargs
            if isinstance(_node_spec, dict):
                raise ValueError(
                    "Error adding function: signature "
                    "cannot contain variable keyword arguments "
                    "(**{})!".format(
                        _arg_name
                    )
                )

            # wrap str in a tuple (normal non-* case)
            if isinstance(_node_spec, str):
                _node_spec = (_node_spec,)

            assert(isinstance(_node_spec, tuple))

            if parameter_namespaces is None:
                parameter_namespaces = ('',) * len(_node_spec)

            # add node or nodes under *args
            for _node_name, _par_namespace in zip(_node_spec, parameter_namespaces):
                _effective_node_name = _par_namespace + _node_name
                # check if parameter node exists
                if _effective_node_name not in self._nodes:
                    if _arg_default == SigParameter.empty:
                        # create empty node and add
                        _pars.append(self.add(Empty(name=_effective_node_name)))
                    else:
                        # create parameter node and add
                        _pars.append(self.add(Parameter(_arg_default, name=_effective_node_name)))
                else:
                    _existing_node = self._nodes[_effective_node_name]

                    # if default value provided
                    if _arg_default != SigParameter.empty:

                        # node is empty
                        if isinstance(_existing_node, Empty):
                            # value encountered for previously empty node -> replace
                            _pars.append(self.add(
                                Parameter(_arg_default, name=_existing_node.name),
                                existing_behavior='replace'
                            ))
                            continue

                        # if node non-empty
                        else:

                            # new default may conflict -> warn if necessary
                            try:
                                assert(_existing_node.value == _arg_default)
                            except AssertionError:
                                # values definitely conflict
                                warnings.warn(
                                    "Ignoring default value {!r} for function parameter '{}': "
                                    "non-empty nexus node already exists and has conflicting "
                                    "value {!r}.".format(
                                        _arg_default,
                                        _effective_node_name,
                                        _existing_node.value
                                    ),
                                    UserWarning
                                )
                            except:
                                # something else may have gone wrong
                                warnings.warn(
                                    "Ignoring default value {!r} for function parameter '{}': "
                                    "non-empty nexus node already exists.".format(
                                        _arg_default,
                                        _effective_node_name
                                    ),
                                    UserWarning
                                )

                    # register existing node as parameter
                    _pars.append(_existing_node)

        # add function node to nexus and return it
        return self.add(
            Function(func, name=func_name, parameters=_pars),
            node_namespace=function_namespace, child_namespaces=parameter_namespaces,
            existing_behavior=existing_behavior
        )

    def add_alias(self, name, alias_for, existing_behavior='fail'):
        """Add an Alias node pointing to an existing node `alias_for`.

        If a node with the same name as the alias node
        has already been added, the behavior of this function can
        be configured via `existing_behavior`. For details, see
        :py:meth:`~kafe.core.fitters.nexus.Nexus.add`.

        :param name: name of the alias node to add
        :type name: str
        :param alias_for: name of the node being pointed to by the alias.
        :type alias_for: str
        :param existing_behavior: how to behave if a node with the same
            name has already been added to the nexus. One of: ``fail``
            (default), ``replace``, ``replace_if_empty``,
            ``replace_if_alias``, ``ignore``.
        :type existing_behavior: str

        :return: the added alias node
        """

        _alias_for_node = self.get(alias_for)
        if _alias_for_node is None:
            raise ValueError(
                "Cannot add alias for node '{}': "
                "node does not exist in nexus!".format(alias_for)
            )

        return self.add(
            Alias(
                _alias_for_node,
                name=name
            ),
            existing_behavior=existing_behavior
        )

    def add_dependency(self, name, depends_on):
        """Register a dependency between nodes explicitly.

        This indicates that the value of node `name` may depend on the
        value of another node `depends_on`. When the value of
        `depends_on` changes, `name` will be notified.

        The specification `depends_on` may be a list or a tuple, in which
        case a dependency is added for each specified node.

        Note that this method should only be used for establishing
        *explicit* dependencies. Other methods such as `add`,
        `add_function` or `add_alias` will create *implicit*
        dependencies between nodes, such as between a `Function` node
        and its parameters or between an `Alias` node and the node it
        points to). In those cases it is not necessary to call
        `add_dependency` explicitly.

        This method is useful when node values may depend on an
        external state that operates on the nexus but is not itself
        part of it.

        :param name: name of the node whose value depends on (an)other node(s)
        :type name: str
        :param depends_on: name(s) of the node(s) whose value node `name` depends on
        :type depends_on: str or tuple/list of str

        :return: self (the Nexus instance)
        :rtype: Nexus
        """
        _node = self.get(name)

        if _node is None:
            raise ValueError(
                "Cannot add dependency: dependent node '{}' does "
                "not exist!".format(name)
            )

        if not (isinstance(depends_on, tuple) or
                isinstance(depends_on, list)):
            depends_on = (depends_on,)


        _not_found = [_dep for _dep in depends_on if self.get(_dep) is None]

        if _not_found:
            raise ValueError(
                "Cannot add dependency: the following nodes passed to "
                "`depends_on` do not exist: {}".format(
                    ', '.join(map(repr, _not_found))
                )
            )

        # add dependent node `name` as a parent of each node in `depends_on`
        for _dep in depends_on:
            self.get(_dep).add_parent(_node)

        # check for cycles
        NodeCycleChecker(self._root_ref()).run()

        return self

    def get(self, node_name):
        """Retrieve a node by its name or ``None`` if no such node exists.

        :param node_name: name of the node to get
        :type node_name: str

        :return: the node or ``None`` if it does not exist
        """
        return self._nodes.get(node_name, None)

    def get_value_dict(self, node_names=None, error_behavior='fail'):
        """Return a mapping of node names to their current values.

        The behavior when encountering an evaluation error can be
        specified as a string:
        *  ``'fail'``: raise an exception
        *  ``'none'``: use ``None`` as a value
        *  ``'exception_as_value'``: use the raised exception as a node value
        *  ``'ignore'``: do not include erroring nodes in result dict
        *  ``'list'``: return list of erroring nodes in result dict
            under special key ``__error__``

        :param node_names: name of the nodes for which to get values.
            If ``None``, all nodes are evaluated.
        :type node_names: list of str or ``None``
        :param error_behavior: how to behave if a node with the same name
            has already been added to the nexus. One of: ``fail`` (default),
            ``none``, ``ignore``, ``list``, ``exception_as_value``.
        :type error_behavior: str

        :return: dict mapping node names to values
        """

        # validate `error_behavior`
        if error_behavior not in self._VALID_GET_DICT_ERROR_BEHAVIORS:
            raise ValueError(
                "Unknown value {!r} for `error_behavior`: "
                "expected one of {!r}".format(
                    error_behavior,
                    self._VALID_GET_DICT_ERROR_BEHAVIORS
                )
            )

        # construct result dict
        _result_dict = {}
        for _name, _node in self._nodes.items():
            # skip root node (has no value anyway)
            if _name == '__root__':
                continue

            # attempt evaluation and behave accordingly on error
            try:
                _val = _node.value
            except Exception as e:
                if error_behavior == 'fail':
                    raise
                elif error_behavior == 'none':
                    _val = None
                elif error_behavior == 'exception_as_value':
                    _val = e
                elif error_behavior == 'ignore':
                    continue
                elif error_behavior == 'list':
                    _result_dict.setdefault('__error__', []).append(_name)
                    continue
                else:
                    assert False  # should not get here
            else:
                _result_dict[_name] = _val

        return _result_dict

    def print_state(self):
        """Print a representation of the nexus state."""
        NodeChildrenPrinter(self._root_ref()).run()

    def source_from(self, other_nexus, namespace=None, namespace_exempt_nodes=None,
                    namespace_exempt_existing_behavior='replace'):
        if namespace is None:
            namespace = ''
        if namespace_exempt_nodes is None:
            namespace_exempt_nodes = []
        for _node in other_nexus._nodes.values():
            if isinstance(_node, ValueNode):
                _child_namespaces = [None if _child.name in namespace_exempt_nodes else namespace
                                     for _child in _node.iter_children()]
                if _node.name in namespace_exempt_nodes:
                    self.add(node=_node, node_namespace=None, child_namespaces=_child_namespaces,
                             existing_behavior=namespace_exempt_existing_behavior)
                else:
                    self.add(node=_node, node_namespace=namespace, child_namespaces=_child_namespaces,
                             existing_behavior='ignore')
            elif isinstance(_node, RootNode):
                pass
            else:
                raise Exception('Unknown node type: %s' % _node.__class__)
