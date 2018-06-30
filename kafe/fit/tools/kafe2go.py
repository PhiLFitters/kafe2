import yaml
from ..histogram import HistFit
from ..indexed import IndexedFit
from ..xy import XYFit

def _construct_fit(fit_type, model_function_kwarg_name, loader, node):
    _kwargs = loader.construct_mapping(node, deep=True)
    __locals_pointer = [None] #TODO better solution?
    _exec_string = _kwargs[model_function_kwarg_name]
    _exec_string += "\n__locals_pointer[0] = locals()"
    exec(_exec_string, {"__builtins__":{"locals":locals}, "__locals_pointer":__locals_pointer})
    _locals = __locals_pointer[0]
    del _locals["__builtins__"]
    del _locals["__locals_pointer"]
    _kwargs[model_function_kwarg_name] = _locals.values()[0] #TODO adjust for multifits
    return fit_type(**_kwargs)

def _construct_hist_fit(loader, node):
    return _construct_fit(HistFit, "model_density_function", loader, node)

def _construct_indexed_fit(loader, node):
    return _construct_fit(IndexedFit, "model_function", loader, node)

def _construct_xy_fit(loader, node):
    return _construct_fit(XYFit, "model_function", loader, node)

def add_constructors():
    yaml.add_constructor("!histfit", _construct_hist_fit)
    yaml.add_constructor("!indexedfit", _construct_indexed_fit)
    yaml.add_constructor("!xyfit", _construct_xy_fit)

def yaml_to_fit(input_string):
    return yaml.load(input_string)

