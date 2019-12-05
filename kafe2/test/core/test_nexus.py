import numpy as np
import unittest2 as unittest

from kafe2.core.fitters.nexus import (
    NodeBase,
    Parameter, Alias,
    Function, Empty,
    Fallback, Tuple, RootNode,
    NodeException, FallbackError,

    NodeChildrenPrinter, NodeCycleChecker,

    _OPERATORS, _UNARY_OPERATORS,

    Nexus, NexusError,
)


class TestNodes(unittest.TestCase):

    DEFAULT_CONSTRUCTIBLE_NODE_TYPES = (
        NodeBase, Empty, RootNode
    )
    PARAMETRIC_NODE_TYPES = (
        Parameter, Alias, Function,
        Fallback, Tuple
    )

    @staticmethod
    def sum_function(a, b):
        return a + b

    def setUp(self):
        self.par_a = Parameter(3)
        self.par_b = Parameter(7)

        self.func_sum_a_b = Function(
            TestNodes.sum_function,
            parameters=(self.par_a, self.par_b)
        )

        self.empty_par = Empty()

    # -- NodeBase

    def test_nodes_default_constructor(self):
        for _node_type in self.DEFAULT_CONSTRUCTIBLE_NODE_TYPES:
            # NodeBase constructor cannot be called because NodeBase has an abstract method.
            if _node_type is not NodeBase:
                _node_type()
                str(_node_type())

    def test_nodes_parametric_constructor(self):
        p = Parameter(None)
        str(p)

        a = Alias(Parameter(None))
        str(a)

        f = Function(lambda x: x, name='_lambda')
        str(f)

        fb = Fallback((f,))
        str(fb)

        t = Tuple([])
        str(t)

    def test_nodes_equality(self):
        p_1 = Parameter(None)
        p_2 = Parameter(None)
        self.assertEqual(p_1, p_1)
        self.assertNotEqual(p_1, p_2)


    def test_nodes_parametric_constructor(self):
        Parameter(None)
        Alias(Parameter(None))
        f = Function(lambda x: x, name='_lambda')
        Fallback((f,))
        Tuple([])

    def test_node_get_children_get_parents(self):
        par_1 = Parameter(3, name='par_1')
        par_2 = Parameter(2, name='par_2')
        par_lc_1 = par_1 + par_2
        par_lc_2 = par_1 - par_2

        for _par in par_1, par_2:
            self.assertEqual(
                _par.get_parents(),
                [par_lc_1, par_lc_2]
            )
        for _par in par_lc_1, par_lc_2:
            self.assertEqual(
                _par.get_children(),
                [par_1, par_2]
            )

    def test_node_iter_children_iter_parents(self):
        par_1 = Parameter(3, name='par_1')
        par_2 = Parameter(2, name='par_2')
        par_lc_1 = par_1 + par_2
        par_lc_2 = par_1 - par_2

        for _par in par_1, par_2:
            self.assertEqual(
                [_parent for _parent in _par.iter_parents()],
                [par_lc_1, par_lc_2]
            )
        for _par in par_lc_1, par_lc_2:
            self.assertEqual(
                [_child for _child in _par.iter_children()],
                [par_1, par_2]
            )

    def test_node_add_child(self):
        par = Parameter(3, name='par')
        par.add_child(1)
        par.add_child(2)

        self.assertEqual(
            [p.value for p in par.iter_children()],
            [1, 2]
        )

    def test_node_add_parent(self):
        par = Parameter(3, name='par')
        par_2 = Parameter(2, name='par_2')

        # Manually adding parents is not allowed:
        with self.assertRaises(NodeException):
            par.add_parent(par_2)

    def test_node_set_children(self):
        par = Parameter(3, name='par')
        child_1 = Parameter(1, name='child_1')
        child_2 = Parameter(2, name='child_2')

        par.set_children([child_1, child_2])

        self.assertEqual(
            par.get_children(),
            [child_1, child_2]
        )

    def test_node_set_parents(self):
        par = Parameter(3, name='par')
        parent_1 = Parameter(1, name='parent_1')
        parent_2 = Parameter(2, name='parent_2')

        par.set_parents([parent_1, parent_2])

        self.assertEqual(
            par.get_parents(),
            [parent_1, parent_2]
        )

    def test_node_remove_child(self):
        par = Parameter(3, name='par')
        child_1 = Parameter(1, name='child_1')
        child_2 = Parameter(2, name='child_2')

        par.set_children([child_1, child_2])
        par.remove_child(child_1)

        self.assertEqual(
            par.get_children(),
            [child_2]
        )

    def test_node_remove_parent(self):
        par = Parameter(3, name='par')
        parent_1 = Parameter(1, name='parent_1')
        parent_2 = Parameter(2, name='parent_2')

        par.set_parents([parent_1, parent_2])
        par.remove_parent(parent_1)

        self.assertEqual(
            par.get_parents(),
            [parent_2]
        )

    def test_parameter_replace(self):
        par = Parameter(3, name='par')
        selector = Function(
            lambda *args: args[1],
            name='selector',
            parameters=(
                Parameter('bla', name='first'),
                par,
                Parameter('blup', name='third'),
            ),
        )
        context = Parameter(2, name='pre_factor') * selector

        # store parents and children for comparison
        _children_before = par.get_children()
        _parents_before = par.get_parents()

        # replace node
        par_new = Parameter(7, name='par_new')
        par.replace(par_new)

        par_new.value = 6

        self.assertEqual(context.value, 12)

        # retrieve new parents and children for comparison
        _children_after = par_new.get_children()
        _parents_after = par_new.get_parents()

        self.assertEqual(_children_before, _children_after)
        self.assertEqual(_parents_before, _parents_after)

    def test_parameter_replace_literal(self):
        par = Parameter(3, name='par')
        selector = Function(
            lambda *args: args[1],
            name='selector',
            parameters=(
                Parameter('bla', name='first'),
                par,
                Parameter('blup', name='third'),
            ),
        )
        context = Parameter(2, name='pre_factor') * selector

        # replace node
        par.replace(7)

        self.assertEqual(context.value, 14)

    def test_function_replace(self):
        par = Parameter(3, name='par')
        selector = Function(
            lambda *args: args[1],
            name='selector',
            parameters=(
                Parameter('bla', name='first'),
                par,
                Parameter('blup', name='third'),
            ),
        )
        context = Parameter(2, name='pre_factor') * selector

        # store parents and children for comparison
        _children_before = selector.get_children()
        _parameters_before = selector.parameters
        _parents_before = selector.get_parents()

        # replace node
        selector_new = Function(lambda *args: args[2], name='selector_new')
        selector.replace(selector_new, other_children=False)

        # retrieve new parents and children for comparison
        _children_after = selector_new.get_children()
        _parents_after = selector_new.get_parents()
        _parameters_after = selector_new.parameters

        self.assertEqual(_children_before, _children_after)
        self.assertEqual(_parents_before, _parents_after)
        self.assertEqual(_parameters_before, _parameters_after)

        self.assertEqual(selector_new.value, 'blup')

    # -- RootNode

    def test_root_node_replace(self):
        r = RootNode()
        with self.assertRaises(TypeError):
            r.replace(7)

    def test_root_node_parents(self):
        r = RootNode()
        with self.assertRaises(TypeError):
            r.add_parent(2)
        self.assertEqual([p for p in r.iter_parents()], [])
        self.assertEqual(r.get_parents(), [])

    # -- Empty

    def test_empty_value(self):
        e = Empty()
        with self.assertRaises(ValueError):
            e.value
        with self.assertRaises(ValueError):
            e.value = 33

    # -- Parameter

    def test_parameter_get_value(self):
        par = Parameter(3)
        self.assertEqual(par.value, 3)

    def test_parameter_set_value(self):
        par = Parameter(3)
        par.value = 7
        self.assertEqual(par.value, 7)

    def test_parameter_name(self):
        par = Parameter(3, name='bla')
        self.assertEqual(par.name, 'bla')

    def test_parameter_invalid_name_raise(self):
        with self.assertRaises(NodeException):
            par = Parameter(3, name='2bla')

    def test_parameter_reserved_name_raise(self):
        for _rn in NodeBase.RESERVED_PARAMETER_NAMES:
            with self.assertRaises(NodeException):
                par = Parameter(3, name=_rn)

    def test_tuple_iter_raise(self):
        par = Parameter(3)
        with self.assertRaises(TypeError):
            iter(par)

    def test_parameter_binary_operation(self):
        par_a = Parameter(3)
        par_b = Parameter(7)
        for _op_name, (_op, _) in _OPERATORS.items():
            par_expr = _op(par_a, par_b)
            self.assertEqual(
                par_expr.value,
                _op(par_a.value, par_b.value)
            )

    def test_parameter_binary_operation_with_literal(self):
        par_a = Parameter(3)
        for _op_name, (_op, _) in _OPERATORS.items():
            par_expr = _op(par_a, 7)
            self.assertEqual(
                par_expr.value,
                _op(par_a.value, 7)
            )
            # reversed order
            par_expr = _op(7, par_a)
            self.assertEqual(
                par_expr.value,
                _op(7, par_a.value)
            )

    def test_parameter_unary_operation(self):
        par = Parameter(3)

        self.assertEqual(
            (+par).value, par.value)
        self.assertEqual(
            (-par).value, -(par.value))
        self.assertEqual(
            (~par).value, ~(par.value))

    # -- Alias

    def test_alias_value(self):
        par = Parameter(3)
        alias = Alias(par, name='alias')
        self.assertEqual(alias.value, par.value)

    def test_multiple_alias_value(self):
        par = Parameter(3)
        alias = Alias(Alias(Alias(par)))
        self.assertEqual(alias.value, par.value)

    def test_alias_ref(self):
        par = Parameter(3)
        alias = Alias(par, name='alias')
        self.assertEqual(alias.ref, par)

    # -- Function

    def test_function_value(self):
        par_a = Parameter(3)
        par_b = Parameter(7)

        def func(a, b):
            return a*10 + b

        func_a_b = Function(func, parameters=(par_a, par_b))

        self.assertEqual(
            func_a_b.value,
            func(par_a.value, par_b.value),
        )

        # test value update
        par_a.value = -3

        self.assertEqual(
            func_a_b.value,
            func(par_a.value, par_b.value),
        )

    def test_function_value_update_frozen(self):
        par_a = Parameter(3)
        par_b = Parameter(7)

        def func(a, b):
            return a*10 + b

        func_a_b = Function(func, parameters=(par_a, par_b))

        self.assertEqual(
            func_a_b.value,
            func(par_a.value, par_b.value),
        )

        # test value update
        func_a_b.freeze()
        par_a.value = -3

        self.assertEqual(
            func_a_b.value,
            func(3, 7),
        )

        func_a_b.unfreeze()
        self.assertEqual(
            func_a_b.value,
            func(par_a.value, par_b.value),
        )

    def test_function_value_lambda(self):
        par_a = Parameter(3)
        par_b = Parameter(7)

        func_a_b = Function(
            lambda a, b: a*10 + b,
            name='_lambda',
            parameters=(par_a, par_b)
        )

        self.assertEqual(
            func_a_b.value,
            func_a_b.func(par_a.value, par_b.value),
        )

    def test_function_parameters(self):
        par_a = Parameter(3)
        par_b = Parameter(7)

        def func(a, b):
            return a*10 + b

        func_a_b = Function(func, parameters=(par_a, par_b))

        self.assertEqual(
            func_a_b.get_children(),
            [par_a, par_b],
        )
        self.assertEqual(
            func_a_b.parameters,
            [par_a, par_b],
        )

    def test_function_auto_parameters(self):
        def func(a, b):
            return a*10 + b

        func_a_b = Function(func)

        par_a = Parameter(3)
        par_b = Parameter(7)

        func_a_b.parameters = [par_a, par_b]

        self.assertEqual(
            func_a_b.parameters,
            [par_a, par_b],
        )

    # -- Fallback

    def test_fallback(self):
        par_a = Parameter(14)
        par_b = Parameter(2)

        div_a_b = Fallback(
            (par_a / par_b, 'DIV0'),
            exception_type=ZeroDivisionError
        )

        # no exception yet
        self.assertEqual(
            div_a_b.value,
            par_a.value / par_b.value,
        )

        # make denominator zero
        par_b.value = 0

        # exception matches type -> fallback
        self.assertEqual(
            div_a_b.value,
            'DIV0',
        )

        # exception does not match type -> raise
        with self.assertRaises(ZeroDivisionError):
            Fallback(
                (par_a / par_b, 'DIV0'),
                exception_type=TypeError
            ).value

    def test_fallback_no_good_alternative(self):
        par_a = Parameter(14)

        div_a_b = Fallback(
            (par_a / 0, 2 * par_a / 0),
            exception_type=ZeroDivisionError
        )

        # no exception yet
        with self.assertRaises(FallbackError):
            div_a_b.value,

    # -- Tuple

    def test_tuple_value(self):
        tuple_a = Tuple((14, 2))
        self.assertEqual(tuple_a.value, (14, 2))

    def test_tuple_len(self):
        tuple_a = Tuple((14, 2))
        self.assertEqual(len(tuple_a), 2)

    def test_tuple_getitem(self):
        tuple_a = Tuple((14, 2))
        self.assertEqual(tuple_a[0].value, 14)
        self.assertEqual(tuple_a[1].value, 2)

    def test_tuple_nodes(self):
        tuple_a = Tuple((14, 2))
        tuple_a.nodes = (9, 4)
        self.assertEqual(tuple_a.value, (9, 4))

    def test_tuple_iter(self):
        tuple_a = Tuple((14, 2))
        self.assertEqual(
            tuple_a.nodes,
            [node for node in tuple_a]
        )

    def test_tuple_iter_values(self):
        tuple_a = Tuple((14, 2))
        self.assertEqual(
            tuple_a.value,
            tuple([value for value in tuple_a.iter_values()])
        )


class TestNodeVisitors(unittest.TestCase):

    # -- NodeCycleChecker

    def test_node_cycle_checker(self):
        par_a = Parameter(1)
        par_b = Parameter(2)
        par_a.set_children([par_b])

        NodeCycleChecker(par_a).run()

        par_b.set_children([par_a])

        with self.assertRaises(ValueError):
            NodeCycleChecker(par_a).run()


class TestNexus(unittest.TestCase):

    def setUp(self):
        self._nexus = Nexus()

    def test_add_get(self):
        par = Parameter('my_value')
        self._nexus.add(par)

        self.assertIs(
            self._nexus.get(par.name),
            par,
        )

    def test_add_get_inexistent(self):
        self.assertIs(
            self._nexus.get('bogus_name'),
            None,
        )

    def test_add_get_explicit_name(self):
        par = Parameter('my_value', name='par')
        self._nexus.add(par)

        self.assertIs(
            self._nexus.get('par'),
            par,
        )

    def test_add_existing_fail(self):
        par = Parameter('my_value')
        self._nexus.add(par)

        with self.assertRaises(NexusError):
            self._nexus.add(par, existing_behavior='fail')

    def test_add_existing_replace(self):
        par_orig = Parameter('my_original_value', name='par')
        self._nexus.add(par_orig)

        par_new = Parameter('my_new_value', name='par')
        self._nexus.add(par_new, existing_behavior='replace')

        self.assertIs(
            self._nexus.get('par'),
            par_new,
        )

    def test_add_existing_ignore(self):
        par_orig = Parameter('my_original_value', name='par')
        self._nexus.add(par_orig)

        par_new = Parameter('my_new_value', name='par')
        self._nexus.add(par_new, existing_behavior='ignore')

        self.assertIs(
            self._nexus.get('par'),
            par_orig,
        )

    def test_add_existing_bogus(self):
        par = Parameter(2)
        self._nexus.add(par)
        with self.assertRaises(ValueError):
            self._nexus.add(par, existing_behavior='bogus')

    def test_add_expression_compare_value(self):
        a = Parameter(3)
        b = Parameter(5)

        expr = (a + b)
        expr.name = 'sum'

        self._nexus.add(a)
        self._nexus.add(b)
        self._nexus.add(expr)

        self.assertEqual(self._nexus.get('sum').value, 8)

    def test_add_expression_test_dependents_added(self):
        a = Parameter(3)
        b = Parameter(5)

        expr = (a - b) / (a + b)
        expr.name = 'asymm'

        expr_value = (a.value - b.value) / (a.value + b.value)

        self._nexus.add(expr)

        self.assertIs(self._nexus.get(a.name), a)
        self.assertIs(self._nexus.get(b.name), b)

    def test_add_function(self):
        def my_func(a, b=2):
            return 2 * a + b

        func_node = self._nexus.add_function(
            func=my_func
        )

        # check if name is as in signature
        self.assertEqual(func_node.name, 'my_func')

        # check if parameter names are as in signature
        self.assertEqual(
            [p.name for p in func_node.parameters],
            ['a', 'b']
        )

        self.assertTrue(isinstance(self._nexus.get('a'), Empty))
        self.assertTrue(isinstance(self._nexus.get('b'), Parameter))
        self.assertEqual(self._nexus.get('b').value, 2)

    def test_add_function_existing_parameters(self):

        a = self._nexus.add(Parameter(4, name='a'))
        b = self._nexus.add(Parameter(5, name='b'))

        def my_func(a, b):
            return 2 * a + b

        func_node = self._nexus.add_function(
            func=my_func
        )

        # check if name is as in signature
        self.assertEqual(func_node.name, 'my_func')

        # check if parameter values correspond to existing parameters
        self.assertEqual(
            [p.value for p in func_node.parameters],
            [a.value, b.value]
        )
        # check function value
        self.assertEqual(func_node.value, my_func(a.value, b.value))

    def test_add_function_existing_parameter_with_default(self):

        a = self._nexus.add(Parameter(4, name='a'))
        b = self._nexus.add(Parameter(9, name='b'))

        def my_func(a=4, b=200):
            return 2 * a + b

        with self.assertWarns(UserWarning) as w:
            func_node = self._nexus.add_function(
                func=my_func
            )

        # check if parameter values correspond to existing parameters
        self.assertEqual(
            [p.value for p in func_node.parameters],
            [a.value, b.value]
        )

        self.assertIn('Ignoring default value', w.warning.args[0])
        self.assertIn('conflicting value', w.warning.args[0])

    def test_add_function_custom_name(self):
        def my_func(a, b):
            return 2 * a + b

        func_node = self._nexus.add_function(
            func=my_func,
            func_name='new_func_name',
        )

        # check if name is as in signature
        self.assertEqual(func_node.name, 'new_func_name')

    def test_add_function_custom_par_names(self):
        x = self._nexus.add(Parameter(7, name='x'))
        y = self._nexus.add(Parameter(8, name='y'))

        def my_func(a, b):
            return 2 * a + b

        func_node = self._nexus.add_function(
            func=my_func,
            par_names=('x', 'y')
        )

        # check function value
        self.assertEqual(func_node.value, my_func(x.value, y.value))

    def test_add_function_varargs(self):
        x = self._nexus.add(Parameter(7, name='x'))
        y = self._nexus.add(Parameter(8, name='y'))

        def my_func(*args):
            return sum(args)

        func_node = self._nexus.add_function(
            func=my_func,
            par_names=('x', 'y')
        )

        # check function value
        self.assertEqual(func_node.value, my_func(x.value, y.value))

    def test_add_detect_cycles(self):
        # define pair of infinitely recursive functions
        def my_rec_func(my_rec_func_2):
            return 2 * my_rec_func_2

        def my_rec_func_2(my_rec_func):
            return 2 * my_rec_func

        func_node = self._nexus.add_function(
            func=my_rec_func,
            par_names=['my_rec_func_2']
        )

        # adding second one triggers cycle detection
        with self.assertRaises(ValueError):
            self._nexus.add_function(
                func=my_rec_func_2,
                existing_behavior='replace',
            )
            func_node.value

    def test_add_dependency(self):
        def test_func(x=3):
            return 2 * x

        func_node = self._nexus.add_function(
            func=test_func,
            par_names=('x',)
        )

        y = self._nexus.add(
            Parameter(4, name='y')
        )

        self.assertEqual(func_node.value, 6)
        self.assertEqual(func_node.stale, False)

        # here 'test_func' does not go stale on 'y' update
        y.value = 8
        self.assertEqual(func_node.stale, False)

        self._nexus.add_dependency('test_func', depends_on='y')

        # now 'test_func' depends on 'y' and should go stale
        y.value = 23
        self.assertEqual(func_node.stale, True)
        self.assertEqual(func_node.value, 6)
