import yaml
import StringIO
import tokenize

from ..histogram import HistFit
from ..indexed import IndexedFit
from ..xy import XYFit

class kafe2goException(Exception):
    pass


def _construct_fit(fit_type, model_function_kwarg_name, loader, node):
    _kwargs = loader.construct_mapping(node, deep=True)
    _kwargs[model_function_kwarg_name] = _parse_model_function(_kwargs[model_function_kwarg_name])
    return fit_type(**_kwargs)

def _construct_hist_fit(loader, node):
    return _construct_fit(HistFit, "model_density_function", loader, node)

def _construct_indexed_fit(loader, node):
    return _construct_fit(IndexedFit, "model_function", loader, node)

def _construct_xy_fit(loader, node):
    return _construct_fit(XYFit, "model_function", loader, node)

def _parse_model_function(_input_string):
    FORBIDDEN_TOKENS = ['eval', 'exec', 'execfile', 'file', 'global', 'import', '__import__', 'input', 
                        'nonlocal', 'open', 'reload', 'self', 'super']
    _tokens = tokenize.generate_tokens(StringIO.StringIO(_input_string).readline)
    for _toknum, _tokval, _spos, _epos, _line_string  in _tokens:
        if _tokval in FORBIDDEN_TOKENS:
            raise kafe2goException("Encountered forbidden token '%s' in user-entered code on line '%s'."
                                    % (_tokval, _line_string))
    
    if "___" in _input_string:
        raise kafe2goException("Model function input must not contain '__'!")
    
    __locals_pointer = [None] #TODO better solution?
    _input_string += "\n__locals_pointer[0] = __locals()"
    exec(_input_string, {"__builtins__":{"__locals":locals}, "__locals_pointer":__locals_pointer})
    _locals = __locals_pointer[0]
    del _locals["__builtins__"]
    del _locals["__locals_pointer"]
    return _locals.values()[0] #TODO adjust for multifits

def add_constructors():
    yaml.add_constructor("!histfit", _construct_hist_fit)
    yaml.add_constructor("!indexedfit", _construct_indexed_fit)
    yaml.add_constructor("!xyfit", _construct_xy_fit)

def yaml_to_fit(input_string):
    return yaml.load(input_string)

