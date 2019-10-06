import unittest2 as unittest
import numpy as np
import scipy.stats

from kafe2.fit.tools.ensemble import (broadcast_to_shape,
                                     EnsembleVariable, EnsembleVariableProbabilityDistribution,
                                     EnsembleError)


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


class TestEnsembleVariable(unittest.TestCase):

    def setUp(self):
        self._ref_size = 1000
        self._ref_shape = (4,7)

        self._ref_total_shape = tuple([self._ref_size] + list(self._ref_shape))

        self._ref_loc = 12.
        self._ref_scale = 3.

        np.random.seed(123456)

        _total_entries = np.prod(self._ref_total_shape)
        self._ref_array = np.random.normal(loc=self._ref_loc, scale=self._ref_scale, size=_total_entries)
        self._ref_array = self._ref_array.reshape(self._ref_total_shape)

        self.ev_no_dist = EnsembleVariable(ensemble_array=self._ref_array,
                                           distribution=None, distribution_parameters=None)
        self.ev_with_dist = EnsembleVariable(ensemble_array=self._ref_array,
                                             distribution=scipy.stats.norm,
                                             distribution_parameters=dict(loc=self._ref_loc, scale=self._ref_scale))

        self._ref_observed_means = np.mean(self._ref_array, axis=0)
        self._ref_observed_stds = np.std(self._ref_array, axis=0)

        self._ref_pdf_frozen = scipy.stats.norm(loc=self._ref_loc, scale=self._ref_scale)
        self._ref_pdf_eval_x = np.linspace(-3, 3, 20)
        self._ref_pdf_eval_y = self._ref_pdf_frozen.pdf(self._ref_pdf_eval_x)
        self._ref_pdf_eval_x_in_shape = np.zeros(self._ref_shape)
        self._ref_pdf_eval_y_in_shape = self._ref_pdf_frozen.pdf(0) * np.ones(self._ref_shape)

    def test_compare_size(self):
        self.assertEqual(self.ev_no_dist.size, self._ref_size)
        self.assertEqual(self.ev_with_dist.size, self._ref_size)

    def test_compare_shape(self):
        self.assertEqual(self.ev_no_dist.shape, self._ref_shape)
        self.assertEqual(self.ev_with_dist.shape, self._ref_shape)

    def test_compare_observed_means(self):
        self.assertTrue(
            np.allclose(
                self.ev_no_dist.mean,
                self._ref_observed_means
            )
        )
        self.assertTrue(
            np.allclose(
                self.ev_with_dist.mean,
                self._ref_observed_means
            )
        )

    def test_compare_observed_stds(self):
        self.assertTrue(
            np.allclose(
                self.ev_no_dist.std,
                self._ref_observed_stds
            )
        )
        self.assertTrue(
            np.allclose(
                self.ev_with_dist.std,
                self._ref_observed_stds
            )
        )

    def test_compare_expected_means(self):
        self.assertTrue(
            np.allclose(
                self.ev_with_dist.dist.mean,
                self._ref_loc
            )
        )

    def test_compare_expected_stds(self):
        self.assertTrue(
            np.allclose(
                self.ev_with_dist.dist.std,
                self._ref_scale
            )
        )

    def test_compare_dist(self):
        self.assertIs(self.ev_no_dist.dist, None)
        self.assertTrue(
            isinstance(self.ev_with_dist.dist, EnsembleVariableProbabilityDistribution)
        )

    def test_compare_dist_eval_pdf_scalar_x(self):
        _eval_y_compare = self.ev_with_dist.dist.eval(0, x_contains_var_shape=False)
        self.assertTrue(
            np.allclose(
                _eval_y_compare,
                self._ref_pdf_frozen.pdf(0)
            )
        )

    def test_compare_dist_eval_pdf_vector_x(self):
        _eval_y_compare = self.ev_with_dist.dist.eval(self._ref_pdf_eval_x, x_contains_var_shape=False)
        self.assertTrue(
            np.allclose(
                self._ref_pdf_eval_y,
                _eval_y_compare
            )
        )

    def test_compare_dist_eval_pdf_in_shape(self):
        _eval_y_compare = self.ev_with_dist.dist.eval(self._ref_pdf_eval_x_in_shape, x_contains_var_shape=True)
        self.assertTrue(
            np.allclose(
                self._ref_pdf_eval_y_in_shape,
                _eval_y_compare
            )
        )
