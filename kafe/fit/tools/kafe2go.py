import yaml
from ..xy import XYFit

def _construct_xy_fit(loader, node):
    _kwargs = loader.construct_mapping(node, deep=True)
    #TODO sanitize
    exec _kwargs["model_function"]
    _kwargs["model_function"] = model_function
    return XYFit(**_kwargs)

def add_constructors():
    yaml.add_constructor("!xyfit", _construct_xy_fit)

def yaml_to_fit(input_string):
    return yaml.load(input_string)

