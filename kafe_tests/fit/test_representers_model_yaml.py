import unittest

TEST_MODEL_FUNCTION_XY="""
model_function:
    python_code: |
        def linear_model(x, a, b):
            return a * x + b
        
"""

class TestXYModelFunctionYamlRepresenter(unittest.TestCase):
    
    def setUp(self):
        pass