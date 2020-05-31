import unittest2 as unittest
from kafe2.test.core.minimizers._base import TestMinimizerMixin
from kafe2.core.minimizers.scipy_optimize_minimizer import MinimizerScipyOptimize


class TestMinimizerScipyOptimize(TestMinimizerMixin, unittest.TestCase):
    def _get_minimizer(self, parameter_names, parameter_values, parameter_errors,
                       function_to_minimize):
        return MinimizerScipyOptimize(
            parameter_names=parameter_names, parameter_values=parameter_values,
            parameter_errors=parameter_errors, function_to_minimize=function_to_minimize
        )

    @property
    def _expected_tolerance(self):
        return 1e-6
