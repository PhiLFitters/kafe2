import unittest
import numpy as np
from kafe.fit.tools import kafe2go

class TestYAMLToFitObject(unittest.TestCase):
    
    def setUp(self):
        kafe2go.add_constructors()
    
    def test_string_to_fit_object(self):
        _input_string="""
        !xyfit
        xy_data: [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]]
        model_function:
            >
            def model_function(x, a , b):
                return a * x + b
        """
        _xy_fit = kafe2go.yaml_to_fit(_input_string)
        _xy_fit.do_fit()
        self.assertTrue(
            np.allclose(
                _xy_fit.parameter_values,
                [2.0, -1.0],
                rtol=1e-3
            )
        )
