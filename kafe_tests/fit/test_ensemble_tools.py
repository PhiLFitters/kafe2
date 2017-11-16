import unittest
import numpy as np

from kafe.fit.tools.ensemble import broadcast_to_shape


class TestCustomBroadcast(unittest.TestCase):

    def setUp(self):
        pass

    def test_compare_shape_scheme_default(self):
        # default numpy broadcasting scheme:
        #        shape:  a x b x c
        #        array:      b x 1
        #      ---------------------
        #       result:  a x b x c
        _array_shape = (4, 6)
        _array = np.empty(_array_shape)
        _target_shape = (5, 3, 4, 6)
        _result_shape = _target_shape
        _broadcasted_array = broadcast_to_shape(_array, _target_shape, scheme='default')
        self.assertEqual(_broadcasted_array.shape, _result_shape)

    def test_compare_shape_scheme_transposed(self):
        # "transposed" numpy broadcasting scheme:
        # (broadcasting is done on the transposed arrays
        #  and the result is transposed back)
        #        shape:  a x b x c
        #        array:  a x 1
        #      ---------------------
        #       result:  a x b x c
        _array_shape = (5, 3)
        _array = np.empty(_array_shape)
        _target_shape = (5, 3, 4, 6)
        _result_shape = _target_shape
        _broadcasted_array = broadcast_to_shape(_array, _target_shape, scheme='transposed')
        self.assertEqual(_broadcasted_array.shape, _result_shape)

    def test_compare_shape_scheme_expand_left(self):
        # `len(shape)` axes are added to the array shape (to the left)
        # and then the 'transposed' broadcasting is applied
        #        shape:  a x b x c          (self.ndim = 3)
        #        array:  1 x 1 x 1 x n x m
        #      -----------------------------
        #       result:  a x b x c x n x m
        _array_shape = (2, 9)
        _array = np.empty(_array_shape)
        _target_shape = (5, 3, 4, 6)
        _result_shape = (5, 3, 4, 6, 2, 9)
        _broadcasted_array = broadcast_to_shape(_array, _target_shape, scheme='expand_left')
        self.assertEqual(_broadcasted_array.shape, _result_shape)

    def test_compare_shape_scheme_expand_right(self):
        # `len(shape)` axes are added to the array shape (to the right)
        # and then broadcasting is done normally
        #        shape:          a x b x c  (self.ndim = 3)
        #        array:  n x m x 1 x 1 x 1
        #      -----------------------------
        #       result:  n x m x a x b x c
        _array_shape = (2, 9)
        _array = np.empty(_array_shape)
        _target_shape = (5, 3, 4, 6)
        _result_shape = (2, 9, 5, 3, 4, 6)
        _broadcasted_array = broadcast_to_shape(_array, _target_shape, scheme='expand_right')
        self.assertEqual(_broadcasted_array.shape, _result_shape)

    def test_compare_shape_scheme_expand_left_successive(self):
        # dimensions are added to the array (on the left) until
        # a broadcastable situation is reached
        # target shape to the array shape:
        #        shape:  a x b
        #        array:      b x n
        #      ---------------------
        #       result:  1 x b x n
        _array_shape = (2, 9, 3)
        _array = np.empty(_array_shape)
        _target_shape = (4, 6, 2, 9)
        _result_shape = (1, 1, 2, 9, 3)
        _broadcasted_array = broadcast_to_shape(_array, _target_shape, scheme='expand_left_successive')
        self.assertEqual(_broadcasted_array.shape, _result_shape)

    def test_compare_shape_scheme_expand_right_successive(self):
        # dimensions are added to the array (on the right) until
        # a broadcastable situation is reached
        # target shape to the array shape:
        #        shape:      a x b
        #        array:  n x a
        #      ---------------------
        #       result:  n x a x 1
        _array_shape = (7, 4, 6)
        _array = np.empty(_array_shape)
        _target_shape = (4, 6, 2, 9)
        _result_shape = (7, 4, 6, 1, 1)
        _broadcasted_array = broadcast_to_shape(_array, _target_shape, scheme='expand_right_successive')
        self.assertEqual(_broadcasted_array.shape, _result_shape)
