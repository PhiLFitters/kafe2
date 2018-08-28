import inspect
import numpy as np
import StringIO
import textwrap
import tokenize
import yaml

from .._base import DReprError, DReprWriterMixin, DReprReaderMixin
from ..container import DataContainerYamlWriter, DataContainerYamlReader
from ..format import ModelFunctionFormatterYamlWriter, ModelFunctionFormatterYamlReader
from ._base import ModelFunctionDReprBase, ParametricModelDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from kafe.fit.xy_multi.model import XYMultiModelFunction, XYMultiParametricModel
from kafe.fit.histogram.model import HistParametricModel
from kafe.fit.indexed.model import IndexedParametricModel
from kafe.fit.xy.model import XYParametricModel

__all__ = ['ModelFunctionYamlWriter', 'ModelFunctionYamlReader', 'ParametricModelYamlWriter', 'ParametricModelYamlReader']

class ModelFunctionYamlWriter(DReprWriterMixin, ModelFunctionDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'writer'

    def __init__(self, model_function, output_io_handle):
        super(ModelFunctionYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_function=model_function)
    
    @staticmethod
    def _make_representation(model_function):
        _yaml_doc = dict()

        # -- determine model function type
        _class = model_function.__class__
        _type = ModelFunctionYamlWriter._MODEL_FUNCTION_CLASS_TO_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Model function unknown or not supported: %s" % _class)
        _yaml_doc['type'] = _type
        _yaml_doc['model_function_formatter'] = ModelFunctionFormatterYamlWriter._make_representation(model_function.formatter)

        if _class is XYMultiModelFunction:
            _python_code_list = []
            for _singular_model_function in model_function.singular_model_functions:
                _python_code = inspect.getsource(_singular_model_function.func)
                _python_code = textwrap.dedent(_python_code) #remove indentation
                _python_code = _python_code.replace("@staticmethod\n","") #remove @staticmethod decorator
                _python_code_list.append(_python_code)
            _yaml_doc['python_code'] = _python_code_list
            _yaml_doc['data_indices'] = model_function.data_indices
        else:
            _python_code = inspect.getsource(model_function.func)
            _python_code = textwrap.dedent(_python_code) #remove indentation
            _python_code = _python_code.replace("@staticmethod\n","") #remove @staticmethod decorator
            #TODO what about other decorators?
            _yaml_doc['python_code'] = _python_code
        
        return _yaml_doc
    
    def write(self):
        self._yaml_doc = self._make_representation(self._model_function)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml_doc, _h, default_flow_style=False)

class ModelFunctionYamlReader(DReprReaderMixin, ModelFunctionDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'reader'
    FORBIDDEN_TOKENS = ['eval', 'exec', 'execfile', 'file', 'global', 'import', '__import__', 'input', 
                        'nonlocal', 'open', 'reload', 'self', 'super']
    
    def __init__(self, input_io_handle):
        super(ModelFunctionYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_function=None)

    @staticmethod
    def _parse_model_function(input_string):
        _tokens = tokenize.generate_tokens(StringIO.StringIO(input_string).readline)
        for _toknum, _tokval, _spos, _epos, _line_string  in _tokens:
            if _tokval in ModelFunctionYamlReader.FORBIDDEN_TOKENS:
                raise DReprError("Encountered forbidden token '%s' in user-entered code on line '%s'."
                                    % (_tokval, _line_string))
    
        if "__" in input_string:
            raise DReprError("Model function input must not contain '__'!")
    
        _fixed_imports = ""
        _model_function_index = 0
        _fixed_imports += "import numpy as np\n" #import numpy
        _model_function_index += 1
        #import scipy if installed
        try:
            import scipy
            _fixed_imports += "import scipy\n"
            _model_function_index += 1
        except:
            pass
        input_string = _fixed_imports + input_string
        
        __locals_pointer = [None] #TODO better solution?
        input_string = input_string + "\n__locals_pointer[0] = __locals()"
        exec(input_string, {"__builtins__":{"__locals":locals, "__import__":__import__}, "__locals_pointer":__locals_pointer})
        _locals = __locals_pointer[0]
        del _locals["__builtins__"]
        del _locals["__locals_pointer"]
        return _locals.values()[_model_function_index] #0 is np
        
    @staticmethod
    def _make_object(yaml_doc):
        # -- determine model function class from type
        _model_function_type = yaml_doc['type']
        _class = ModelFunctionYamlReader._MODEL_FUNCTION_TYPE_NAME_TO_CLASS.get(_model_function_type, None)
        if _class is None:
            raise DReprError("Model function type unknown or not supported: {}".format(_model_function_type))
        
        if _class is XYMultiModelFunction:
            _python_code_list = yaml_doc["python_code"]
            if isinstance(_python_code_list, str): #check if only one model function is given
                _python_code_list = [_python_code_list]
            _python_function_list = [
                ModelFunctionYamlReader._parse_model_function(_python_code)
                for _python_code in _python_code_list
            ]
            _data_indices = yaml_doc["data_indices"]
            _model_function_object = XYMultiModelFunction(_python_function_list, _data_indices)
        else:
            _python_function = ModelFunctionYamlReader._parse_model_function(yaml_doc["python_code"])
            _model_function_object = _class(_python_function)
        
        #construct model function formatter if specified
        if 'model_function_formatter' in yaml_doc:
            _model_function_object._formatter = ModelFunctionFormatterYamlReader._make_object(yaml_doc['model_function_formatter'])
        
        return _model_function_object
    
    def read(self):
        with self._ihandle as _h:
            self._yaml_doc = yaml.load(_h)
        return self._make_object(self._yaml_doc)

class ParametricModelYamlWriter(DReprWriterMixin, ParametricModelDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'writer'

    def __init__(self, parametric_model, output_io_handle):
        super(ParametricModelYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            parametric_model=parametric_model)
    
    @staticmethod
    def _make_representation(parametric_model):
        _yaml_doc = dict()

        # -- determine model function type
        _class = parametric_model.__class__
        _type = ParametricModelYamlWriter._PARAMETRIC_MODEL_CLASS_TO_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Parametric model unknown or not supported: %s" % _class)
        _yaml_doc['type'] = _type
        
        # -- write representation for model types
        if _class is HistParametricModel:
            _yaml_doc['n_bins'] = parametric_model.n_bins
            _yaml_doc['bin_range'] = parametric_model.bin_range
            _yaml_doc['model_density_function'] = ModelFunctionYamlWriter._make_representation(
                parametric_model._model_function_object)
            _yaml_doc['bin_edges'] = parametric_model.bin_edges.tolist()
            _yaml_doc['model_density_func_antiderivative'] = None #TODO implement
        elif _class is IndexedParametricModel:
            _yaml_doc['data'] = parametric_model.data.tolist()
            _yaml_doc['model_function'] = ModelFunctionYamlWriter._make_representation(
                parametric_model._model_function_object)
        elif _class is XYParametricModel:
            _yaml_doc['x_data'] = parametric_model.x.tolist()
            _yaml_doc['y_data'] = parametric_model.y.tolist()
            _yaml_doc['model_function'] = ModelFunctionYamlWriter._make_representation(
                parametric_model._model_function_object)
        elif _class is XYMultiParametricModel:
            for _i in range(parametric_model.num_datasets):
                _yaml_doc['x_data_%s' % _i] = parametric_model.get_splice(parametric_model.x, _i).tolist()
                _yaml_doc['y_data_%s' % _i] = parametric_model.get_splice(parametric_model.y, _i).tolist()
                _yaml_doc['model_function'] = ModelFunctionYamlWriter._make_representation(
                    parametric_model._model_function_object)
        
        _parameters = parametric_model.parameters
        if isinstance(_parameters, np.ndarray):
            _parameters = _parameters.tolist() #better readability in file
        _yaml_doc['model_parameters'] = _parameters

        # -- write error representation for all container types
        if parametric_model.has_errors:
            DataContainerYamlWriter._write_errors_to_yaml(parametric_model, _yaml_doc)
        
        
        return _yaml_doc
    
    def write(self):
        self._yaml_doc = self._make_representation(self._parametric_model)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml_doc, _h, default_flow_style=False)

class ParametricModelYamlReader(DReprReaderMixin, ParametricModelDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'reader'
    
    def __init__(self, input_io_handle):
        super(ParametricModelYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            parametric_model=None)

        
    @staticmethod
    def _make_object(yaml_doc):
        # -- determine model function class from type
        _parametric_model_type = yaml_doc['type']
        _class = ParametricModelYamlReader._PARAMETRIC_MODEL_TYPE_NAME_TO_CLASS.get(_parametric_model_type, None)
        if _class is None:
            raise DReprError("Model function type unknown or not supported: {}".format(_parametric_model_type))

        #_kwarg_list = kwargs that directly correspond to unchanging scalar yaml entries
        _kwarg_list = ['model_parameters']         
        _constructor_kwargs = {}
        if _class is HistParametricModel:
            _kwarg_list.append('n_bins')
            _kwarg_list.append('bin_range')
            _kwarg_list.append('bin_edges')
            #TODO implement parsing
            _kwarg_list.append('model_density_func_antiderivative')
        elif _class is IndexedParametricModel:
            _kwarg_list.append('shape_like')
        elif _class is XYParametricModel:
            _kwarg_list.append('x_data')
        elif _class is XYMultiParametricModel:
            _x_data = []
            _i = 0 #xy dataset index
            _x_data_i = yaml_doc.get('x_data_%s' % _i, None)
            # y data not read, only written for human understanding
            #_y_data_i = yaml_doc.get('y_data_%s' % _i, None)
            while _x_data_i is not None:
                _x_data += _x_data_i
                _i += 1
                _x_data_i = yaml_doc.get('x_data_%s' % _i, None)
                #_y_data_i = yaml_doc.get('y_data_%s' % _i, None)
            _constructor_kwargs['x_data'] = _x_data
        _constructor_kwargs.update({key: yaml_doc.get(key, None) for key in _kwarg_list})
        if _class is HistParametricModel:
            _model_density_func = ModelFunctionYamlReader._make_object(yaml_doc['model_density_function'])
            _model_density_func.defaults = _constructor_kwargs['model_parameters']
            _constructor_kwargs['model_density_func'] = _model_density_func
        elif _class in (IndexedParametricModel, XYParametricModel, XYMultiParametricModel):
            _model_func = ModelFunctionYamlReader._make_object(yaml_doc['model_function'])
            _model_func.defaults = _constructor_kwargs['model_parameters']
            _constructor_kwargs['model_func'] = _model_func
        _parametric_model_object = _class(**_constructor_kwargs)
        
        # -- process error sources
        if _class in (XYParametricModel, XYMultiParametricModel):
            _xerrs = yaml_doc.get('x_errors', [])
            _yerrs = yaml_doc.get('y_errors', [])
            _errs = _xerrs + _yerrs
            _axes = [0] * len(_xerrs) + [1] * len(_yerrs)  # 0 for 'x', 1 for 'y'
        else:
            _errs = yaml_doc.get('errors', [])
            _axes = [None] * len(_errs)

        # add error sources, if any
        for _err, _axis in zip(_errs, _axes):
            _add_kwargs = dict()
            # translate and check that all required keys are present
            try:
                _err_type = _err['type']

                _add_kwargs['name'] = _err['name']

                if _err_type == 'simple':
                    _add_kwargs['err_val']= _err['error_value']
                    _add_kwargs['correlation']= _err['correlation_coefficient']
                elif _err_type == 'matrix':
                    _add_kwargs['err_matrix'] = _err['matrix']
                    _add_kwargs['matrix_type'] = _err['matrix_type']
                    _add_kwargs['err_val'] = _err.get('error_value', None)  # only mandatory for cor mats; check done later
                else:
                    raise DReprError("Unknown error type '{}'. "
                                     "Valid: {}".format(_err_type, ('simple', 'matrix')))

                _add_kwargs['relative'] = _err['relative']

                # if needed, specify the axis (only for 'xy' containers)
                if _axis is not None:
                    _add_kwargs['axis'] = _axis
            except KeyError as e:
                # KeyErrors mean the YAML is incomplete -> raise
                raise DReprError("Missing required key '%s' for error specification" % e.args[0])

            # add error to parametric model
            DataContainerYamlReader._add_error_to_container(_err_type, _parametric_model_object, **_add_kwargs)
        
        return _parametric_model_object
    
    def read(self):
        with self._ihandle as _h:
            self.yaml_doc = yaml.load(_h)
        return self._make_object(self.yaml_doc)

# register the above classes in the module-level dictionary
ModelFunctionYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
ParametricModelYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ParametricModelYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
