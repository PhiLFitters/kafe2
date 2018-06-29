import yaml
from ..xy import XYFit

def _construct_xy_fit(loader, node):
    _kwargs = loader.construct_mapping(node, deep=True)
    __locals_pointer = [None] #TODO better solution?
    _exec_string = _kwargs["model_function"]
    _exec_string += "\n__locals_pointer[0] = locals()"
    exec(_exec_string, {"__builtins__":{"locals":locals}, "__locals_pointer":__locals_pointer})
    _locals = __locals_pointer[0]
    del _locals["__builtins__"]
    del _locals["__locals_pointer"]
    _kwargs["model_function"] = _locals.values()[0]
    return XYFit(**_kwargs)

def add_constructors():
    yaml.add_constructor("!xyfit", _construct_xy_fit)

def yaml_to_fit(input_string):
    return yaml.load(input_string)

