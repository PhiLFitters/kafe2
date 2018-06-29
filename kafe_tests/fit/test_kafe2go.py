import unittest
import numpy as np
import logging
import sys
from kafe.fit.tools import kafe2go

class TestYAMLToFitObject(unittest.TestCase):
    
    def setUp(self):
        logger = logging.getLogger()
        logger.level = logging.DEBUG
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        kafe2go.add_constructors()
    
    def test_string_to_fit_object(self):
        _input_string="""
        !xyfit
        xy_data: [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]]
        model_function:
            >
            def linear_model(x, a , b):
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
