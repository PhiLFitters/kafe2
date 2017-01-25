import unittest

from ..parameter_space import ParameterSpaceException, ParameterException, ParameterSpace


class TestPSpace(unittest.TestCase):

    def setUp(self):
        ps = ParameterSpace()
        ps.set('a1', 62, create=True)
        ps.set('b1', 63, create=True)
        ps.set('c1', 64, create=True)
        ps.new_alias('a2', 'a1')
        ps.new_alias('b2', 'b1')
        self.ps = ps

    def test_raise_create_invalid_parameter_name(self):
        with self.assertRaises(ParameterSpaceException):
            self.ps.set('*_invalid', 42., create=True)

    def test_raise_alias_invalid_parameter_name(self):
        with self.assertRaises(ParameterSpaceException):
            self.ps.new_alias('*_invalid', 'a1')

    def test_raise_set_invalid_parameter_name(self):
        with self.assertRaises(ParameterException):
            self.ps.get('a1').name = '*_invalid'

    def test_raise_create_reserved_parameter_name(self):
        with self.assertRaises(ParameterSpaceException):
            self.ps.set('__all__', 42., create=True)
        with self.assertRaises(ParameterSpaceException):
            self.ps.set('__real__', 42., create=True)

    def test_raise_alias_reserved_parameter_name(self):
        with self.assertRaises(ParameterSpaceException):
            self.ps.new_alias('__all__', 'a1')
        with self.assertRaises(ParameterSpaceException):
            self.ps.new_alias('__real__', 'a1')

    def test_raise_set_reserved_parameter_name(self):
        with self.assertRaises(ParameterException):
            self.ps.get('a1').name = '__all__'
        with self.assertRaises(ParameterException):
            self.ps.get('a1').name = '__real__'

    def test_get_nonexistent(self):
        self.assertIsNone(self.ps.get('*z9'))

    def test_get_nonexistent(self):
        self.assertIsNone(self.ps.get('*z9'))

    def test_get_existent(self):
        self.assertEqual(self.ps.get('a1').value, 62)
        self.assertEqual(self.ps.get('b1').value, 63)
        self.assertEqual(self.ps.get('c1').value, 64)

    def test_get_all_real(self):
        self.assertEqual(self.ps.get_values("__real__"), [62, 63, 64])

    def test_alias(self):
        self.assertIs(self.ps.get('a1'), self.ps.get('a2'))
        self.assertIs(self.ps.get('b1'), self.ps.get('b2'))

    def test_set_inexistent_with_create_false(self):
        with self.assertRaises(ParameterSpaceException) as ve:
            self.ps.set('*z1', 44, create=False)

    def test_set_existent_with_create_true(self):
        with self.assertRaises(ParameterSpaceException) as ve:
            self.ps.set('a1', 44, create=True)

    def test_new_alias_par_doesnt_exist(self):
        self.assertRaises(ParameterSpaceException,
                          self.ps.new_alias,
                          '*z2',
                          '*z1')

    def test_new_alias_alias_exists(self):
        self.assertRaises(ParameterSpaceException,
                          self.ps.new_alias,
                          'a2',
                          'a1')

    def test_ids(self):
        self.assertDictEqual(self.ps.ids,
                             {'a1': 0, 'c1': 2, 'a2': 0, 'b1': 1, 'b2': 1})

    def test_dim(self):
        self.assertEqual(self.ps.dimension, 3)
