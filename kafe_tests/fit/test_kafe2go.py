import unittest
import numpy as np
import logging
import sys
from kafe.fit.tools import kafe2go, kafe2goException
from kafe.fit import HistFit, IndexedFit, XYFit

class TestYAMLToFitObject(unittest.TestCase):
    
    def setUp(self):
        logger = logging.getLogger()
        logger.level = logging.DEBUG
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        kafe2go.add_constructors()
        self._input_string_hist="""
        !histfit
        data: [-0.32523704 -0.83835927  0.13349729 -1.63891736  0.62477429  0.44002838
                1.57854108 -2.26540303 -2.26564588 -0.65707497  0.54147452 -1.14683608
               -0.40344333  1.19959957  0.94797447 -0.62652507  0.77217068  0.86948057
               -0.63931895  0.80411386]
        model_density_function:
            >
            def linear_model(x, a , b):
                return a * x + b
        """
        self._input_string_indexed="""
        !indexedfit
        data: [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]
        model_function:
            >
            def linear_model(x, a , b):
                return a * x + b
        """
        self._input_string_xy="""
        !xyfit
        xy_data: [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]]
        model_function:
            >
            def linear_model(x, a , b):
                return a * x + b
        """

    
    def test_string_to_hist_fit(self):
        _hist_fit = kafe2go.yaml_to_fit(self._input_string_hist)
        self.assertTrue(isinstance(_hist_fit, HistFit))
    
    def test_string_to_indexed_fit(self):
        _indexed_fit = kafe2go.yaml_to_fit(self._input_string_indexed)
        print _indexed_fit
        self.assertTrue(isinstance(_indexed_fit, IndexedFit))
    
    def test_string_to_xy_fit(self):
        _xy_fit = kafe2go.yaml_to_fit(self._input_string_xy)
        self.assertTrue(isinstance(_xy_fit, XYFit))
    
    def test_string_to_xy_fit_do_fit(self):
        _xy_fit = kafe2go.yaml_to_fit(self._input_string_xy)
        _xy_fit.do_fit()
        self.assertTrue(
            np.allclose(
                _xy_fit.parameter_values,
                [2.0, -1.0],
                rtol=1e-3
            )
        )

class TestInputSanitization(unittest.TestCase):
    
    def setUp(self):
        kafe2go.add_constructors()
        self._input_string_forbidden_token="""
        !xyfit
        xy_data: [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]]
        model_function:
            >
            def linear_model(x, a , b):
                import os
                return a * x + b
        """
    
    def test_forbidden_token(self):
        with self.assertRaises(kafe2goException):
            kafe2go.yaml_to_fit(self._input_string_forbidden_token)
            