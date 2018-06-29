import unittest
from kafe.fit.tools import yaml_to_fit

class TestYAMLToFitObject(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_string_to_fit_object(self):
        input_string="""
        !!python/object:kafe.fit.XYFit
        xy_data: 0.0
        model_function: 0.0
        """
        yaml_to_fit(input_string)
