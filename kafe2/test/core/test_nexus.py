import unittest2 as unittest

from kafe2.core.fitters import Nexus
from kafe2.core.fitters.nexus import NodeException, NexusException, NODE_VALUE_DEFAULT

class TestParameterFunction(unittest.TestCase):
    @staticmethod
    def native_func_difference12(a1, a2):
        return a1 - a2

    @staticmethod
    def native_func_difference13(a1, a3):
        return a1 - a3

    @staticmethod
    def native_func_difference3_constant2(a3):
        return a3 - 2

    @staticmethod
    def native_func_difference1_constant2(a1):
        return a1 - 2

    @staticmethod
    def native_func_unknown_parameters(a1, zz_unknown_zz):
        return a1 - zz_unknown_zz

    def setUp(self):
        self.ps = Nexus()
        self.ps.new(a1=62)
        self.ps.new(a2=42)
        self.ps.new_alias(a3='a1')

        self.ps.new_function(self.native_func_difference12)
        self.ps.new_function(self.native_func_difference13)
        self.ps.new_function(self.native_func_difference3_constant2)
        self.ps.new_function(self.native_func_difference1_constant2)


    def test_value_func_difference12(self):
        self.assertEqual(self.ps.get_values('native_func_difference12'), 20)

    def test_value_func_difference13(self):
        self.assertEqual(self.ps.get_values('native_func_difference13'), 0)

    def test_value_func_difference3_constant2(self):
        self.assertEqual(self.ps.get_values('native_func_difference3_constant2'), 60)

    def test_update_func_value_from_parameter(self):
        # print '-- self.assertEqual(self.ps.get_values(\'native_func_difference1_constant2\'), 60) --'
        self.assertEqual(self.ps.get_values('native_func_difference1_constant2'), 60)
        # print '== self.ps.set(a3=42) ==============================================================='
        self.ps.set(a1=42)
        # print '++ self.assertEqual(self.ps.get_values(\'native_func_difference1_constant2\'), 40) ++'
        self.assertEqual(self.ps.get_values('native_func_difference1_constant2'), 40)

    def test_update_func_value_from_parameter_alias(self):
        # print '-- self.assertEqual(self.ps.get_values(\'native_func_difference3_constant2\'), 60) --'
        self.assertEqual(self.ps.get_values('native_func_difference3_constant2'), 60)
        # print '== self.ps.set(a3=42) ==============================================================='
        self.ps.set(a3=42)
        # print '++ self.assertEqual(self.ps.get_values(\'native_func_difference3_constant2\'), 40) ++'
        self.assertEqual(self.ps.get_values('native_func_difference3_constant2'), 40)

    def test_raise_register_func_with_unknown_parameters(self):
        with self.assertRaises(NexusException):
            self.ps.new_function(self.native_func_unknown_parameters)

    def test_check_noraise_register_func_with_unknown_parameters(self):
        self.ps.new_function(self.native_func_unknown_parameters, add_unknown_parameters=True)

    def test_check_success_register_func_with_unknown_parameters(self):
        self.ps.new_function(self.native_func_unknown_parameters, add_unknown_parameters=True)
        self.assertEqual(self.ps.get_by_name('zz_unknown_zz').value, NODE_VALUE_DEFAULT)


class TestPSpace(unittest.TestCase):

    def setUp(self):
        ps = Nexus()
        ps.new(a1=62)
        ps.new(b1=63)
        ps.new(c1=64)
        ps.new_alias(a2='a1')
        ps.new_alias(b2='b1')
        self.ps = ps

    def test_parent_pspace_is_ref_pspace(self):
        self.assertIs(self.ps.get('a1').nexus, self.ps)

    def test_raise_set_invalid_parameter_name(self):
        with self.assertRaises(NodeException):
            self.ps.get('a1').name = '*_invalid'

    def test_raise_create_reserved_parameter_name(self):
        with self.assertRaises(NexusException):
            self.ps.new(__all__=42.)
        #with self.assertRaises(ParameterSpaceException):
            self.ps.new(__real__=42.)

    def test_raise_alias_reserved_parameter_name(self):
        with self.assertRaises(NexusException):
            self.ps.new_alias(__all__='a1')
        with self.assertRaises(NexusException):
            self.ps.new_alias(__real__='a1')

    def test_raise_set_reserved_parameter_name(self):
        with self.assertRaises(NodeException):
            self.ps.get('a1').name = '__all__'
        with self.assertRaises(NodeException):
            self.ps.get('a1').name = '__real__'

    def test_get_nonexistent(self):
        self.assertIsNone(self.ps.get('*z9'))

    def test_get_existent(self):
        self.assertEqual(self.ps.get('a1').value, 62)
        self.assertEqual(self.ps.get('b1').value, 63)
        self.assertEqual(self.ps.get('c1').value, 64)

    def test_get_all_real(self):
        self.assertEqual(self.ps.get_values("__real__"), [62, 63, 64])

    def test_get_all(self):
        self.assertEqual(self.ps.get_values("__all__"), [62, 63, 64, 62, 63])

    def test_get_by_name_nonexistent(self):
        self.assertIsNone(self.ps.get_by_name('*z9'))

    def test_get_by_name_existent(self):
        self.assertEqual(self.ps.get_by_name('a1').value, 62)
        self.assertEqual(self.ps.get_by_name('b1').value, 63)
        self.assertEqual(self.ps.get_by_name('c1').value, 64)

    def test_alias(self):
        self.assertIs(self.ps.get('a1'), self.ps.get('a2'))
        self.assertIs(self.ps.get('b1'), self.ps.get('b2'))

    def test_alias_get_by_name(self):
        self.assertIs(self.ps.get_by_name('a1'), self.ps.get_by_name('a2'))
        self.assertIs(self.ps.get_by_name('b1'), self.ps.get_by_name('b2'))

    def test_compare_get_get_by_name(self):
        for _pn in self.ps.parameter_names:
            self.assertIs(self.ps.get_by_name(_pn), self.ps.get(_pn))

    def test_raise_set_parameter_not_exists(self):
        with self.assertRaises(NexusException) as ve:
            self.ps.set(inexistent=44)

    def test_raise_new_parameter_exists(self):
        with self.assertRaises(NexusException) as ve:
            self.ps.new(a1=44)

    def test_raise_new_alias_par_doesnt_exist(self):
        with self.assertRaises(NexusException):
            self.ps.new_alias(z2='z1')

    def test_raise_new_alias_alias_exists(self):
        with self.assertRaises(NexusException):
            self.ps.new_alias(a2='a1')

    def test_dim(self):
        self.assertEqual(self.ps.dimension, 3)


# class TestPSpaceFunctionality(unittest.TestCase):
#
#     def setUp(self):
#         self.ps = ParameterSpace()
#         self.ps.new(a1=62)
#         self.ps.new(a2=42)
#         self.ps.new_alias(a3='a1')
#
#         def native_func_difference12(a1, a2):
#             return a1 - a2
#
#         def native_func_difference13(a1, a3):
#             return a1 - a3
#
#         self.native_func_difference12 = native_func_difference12
#         self.native_func_difference13 = native_func_difference13
#
#         self.ps.new_function(self.native_func_difference12)
#         self.ps.new_function(self.native_func_difference13)
#
#     def test_value_func_difference12(self):
#         pass
#
#     def test_value_func_difference13(self):
#         pass


