import inspect
import numpy as np
import six
import textwrap
import tokenize

from .._base import DReprError
from .._yaml_base import YamlWriterMixin, YamlReaderMixin
from ..container import DataContainerYamlWriter, DataContainerYamlReader
from ..format import ModelFunctionFormatterYamlWriter, ModelFunctionFormatterYamlReader
from ._base import ModelFunctionDReprBase, ParametricModelDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from ....fit import (
    HistModelFunction, HistParametricModel, IndexedParametricModel, IndexedModelFunction, 
    XYParametricModel, XYModelFunction, XYMultiModelFunction, XYMultiParametricModel
)
from .._yaml_base import YamlWriterException, YamlReaderException
from ....fit.util import function_library

__all__ = ['ModelFunctionYamlWriter', 'ModelFunctionYamlReader', 'ParametricModelYamlWriter', 'ParametricModelYamlReader']

class ModelFunctionYamlWriter(YamlWriterMixin, ModelFunctionDReprBase):

    def __init__(self, model_function, output_io_handle):
        super(ModelFunctionYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            model_function=model_function)
    
    @classmethod
    def _make_representation(cls, model_function):
        _yaml_doc = dict()

        # -- determine model function type
        _class = model_function.__class__
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Model function type unknown or not supported: %s" % _class)
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
            _python_code = model_function.source_code
            _python_code = textwrap.dedent(_python_code) #remove indentation
            _python_code = _python_code.replace("@staticmethod\n","") #remove @staticmethod decorator
            #TODO what about other decorators?
            _yaml_doc['python_code'] = _python_code
        
        return _yaml_doc
    
class ModelFunctionYamlReader(YamlReaderMixin, ModelFunctionDReprBase):

    FORBIDDEN_TOKENS = ['compile', 'eval', 'exec', 'execfile', 'file', 'global', 'globals', 'import', '__import__', 
                        'input', ' locals', 'nonlocal', 'open', 'reload', 'self', 'super']
    
    def __init__(self, input_io_handle):
        super(ModelFunctionYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_function=None)

    @staticmethod
    def _parse_model_function(input_string):
        """converts a string of python code into a python function object"""
        _tokens = tokenize.generate_tokens(six.StringIO(input_string).readline)
        for _toknum, _tokval, _spos, _epos, _line_string  in _tokens:
            if _tokval in ModelFunctionYamlReader.FORBIDDEN_TOKENS:
                raise DReprError("Encountered forbidden token '%s' in user-entered code on line '%s'."
                                    % (_tokval, _line_string))
    
        if "__" in input_string:
            raise DReprError("Model function input must not contain '__'!")
    
        _imports = ""
        _imports += "import numpy as np\n" #import numpy
        #import scipy if installed
        try:
            import scipy
            _imports += "import scipy\n"
        except:
            pass
        
        
        __locals_pointer = [None, None] #TODO better solution?
        #save locals before function definition
        _exec_string = _imports + "__locals_pointer[0] = __locals().copy()\n" + input_string
        #save locals after function definition
        _exec_string = _exec_string + "\n__locals_pointer[1] = __locals().copy()"
        _restricted_builtins = __builtins__.copy()
        for _forbidden_token in ModelFunctionYamlReader.FORBIDDEN_TOKENS:
            _restricted_builtins.pop(_forbidden_token, None)
        _restricted_builtins['__import__'] = __import__
        exec(_exec_string, {"__builtins__":_restricted_builtins, "__locals":locals, 
                            "__locals_pointer":__locals_pointer})
        _locals_pre, _locals_post = __locals_pointer[0].values(), __locals_pointer[1].values()
        _new_references = []
        for _post_reference in _locals_post:
            if _post_reference not in _locals_pre:
                _new_references.append(_post_reference)
        if len(_new_references) != 1:
            raise YamlReaderException(
                "Expected to receive exactly one new reference as a model function but instead received %s in the following string:\n%s"
                % (len(_new_references), input_string))
        return _new_references[0]
    
    @classmethod
    def _get_subspace_override_dict(cls, model_function_class):
        _override_dict = {'arg_formatters':'model_function_formatter'}

        if model_function_class is HistModelFunction:
            _override_dict['model_density_function_name'] = 'model_function_formatter'
            _override_dict['latex_model_density_function_name'] = 'model_function_formatter'
            _override_dict['x_name'] = 'model_function_formatter'
            _override_dict['latex_x_name'] = 'model_function_formatter'
            _override_dict['expression_string'] = 'model_function_formatter'
            _override_dict['latex_expression_string'] = 'model_function_formatter'
        elif model_function_class is IndexedModelFunction:
            _override_dict['model_function_name'] = 'model_function_formatter'
            _override_dict['latex_model_function_name'] = 'model_function_formatter'
            _override_dict['index_name'] = 'model_function_formatter'
            _override_dict['latex_index_name'] = 'model_function_formatter'
            _override_dict['expression_string'] = 'model_function_formatter'
            _override_dict['latex_expression_string'] = 'model_function_formatter'
        elif model_function_class is XYModelFunction:
            _override_dict['model_function_name'] = 'model_function_formatter'
            _override_dict['latex_model_function_name'] = 'model_function_formatter'
            _override_dict['x_name'] = 'model_function_formatter'
            _override_dict['latex_x_name'] = 'model_function_formatter'
            _override_dict['expression_string'] = 'model_function_formatter'
            _override_dict['latex_expression_string'] = 'model_function_formatter'
        elif model_function_class is XYMultiModelFunction:
            for _i in range(10):
                _override_dict['model_function_name_%s' % _i] = 'model_function_formatter'
                _override_dict['latex_model_function_name_%s' % _i] = 'model_function_formatter'
                _override_dict['x_name_%s' % _i] = 'model_function_formatter'
                _override_dict['latex_x_name_%s' % _i] = 'model_function_formatter'
                _override_dict['expression_string_%s' % _i] = 'model_function_formatter'
                _override_dict['latex_expression_string_%s' % _i] = 'model_function_formatter'
            _override_dict['x_name'] = 'model_function'
            _override_dict['latex_x_name'] = 'model_function'
        else:
            raise YamlReaderException("Unknown model function class: %s" % model_function_class)
        return _override_dict

    @classmethod
    def _get_required_keywords(cls, yaml_doc, model_function_class):
        if model_function_class in (HistModelFunction, IndexedModelFunction, XYModelFunction):
            return ['python_code']
        elif model_function_class is XYMultiModelFunction:
            return ['python_code', 'data_indices']
        else:
            raise YamlReaderException("Unknown model function class: %s" % model_function_class)
        
    @classmethod
    def _process_string(cls, string_representation, default_type):
        return dict(type=default_type, python_code=string_representation)
    
    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine model function class from type
        _model_function_type = yaml_doc.pop('type')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_model_function_type)
        
        if _class is XYMultiModelFunction:
            _python_code_list = yaml_doc.pop("python_code")
            if isinstance(_python_code_list, str): #check if only one model function is given
                _python_code_list = [_python_code_list]
            _python_function_list = []
            for _python_code in _python_code_list:
                _python_function = ModelFunctionYamlReader._parse_model_function(_python_code)
                _python_function._source_code = _python_code
                _python_function_list.append(_python_function)
            _data_indices = yaml_doc.pop("data_indices")
            _model_function_object = XYMultiModelFunction(_python_function_list, _data_indices)
        else:
            _raw_string = yaml_doc.pop("python_code")
            _function_library_entry = function_library.STRING_TO_FUNCTION.get(_raw_string, None)
            if _function_library_entry:
                _model_function_object = _class(_function_library_entry)
            else:
                _parsed_function = ModelFunctionYamlReader._parse_model_function(_raw_string)
                _model_function_object = _class(_parsed_function)
                _model_function_object._source_code = _raw_string
        
        #construct model function formatter if specified
        _model_function_formatter_yaml = yaml_doc.pop('model_function_formatter', None)
        if _model_function_formatter_yaml:
            _model_function_object._formatter = ModelFunctionFormatterYamlReader._make_object(
                _model_function_formatter_yaml)
        
        return _model_function_object, yaml_doc
    
class ParametricModelYamlWriter(YamlWriterMixin, ParametricModelDReprBase):

    def __init__(self, parametric_model, output_io_handle):
        super(ParametricModelYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            parametric_model=parametric_model)
    
    @classmethod
    def _make_representation(cls, parametric_model):
        _yaml_doc = dict()

        # -- determine model function type
        _class = parametric_model.__class__
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Parametric model type unknown or not supported: %s" % _class)
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
            _yaml_doc['shape_like'] = parametric_model.data.tolist()
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
        else:
            raise YamlWriterException("Unkonwn parametric model type")

        # convert all numpy array entries to regular float, then convert to list, improves readability
        _yaml_doc['model_parameters'] = np.array(parametric_model.parameters, dtype=float).tolist()

        # -- write error representation for all container types
        if parametric_model.has_errors:
            DataContainerYamlWriter._write_errors_to_yaml(parametric_model, _yaml_doc)
        
        
        return _yaml_doc
    
class ParametricModelYamlReader(YamlReaderMixin, ParametricModelDReprBase):
    
    def __init__(self, input_io_handle):
        super(ParametricModelYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            parametric_model=None)

    @classmethod
    def _get_subspace_override_dict(cls, parametric_model_class):
        _override_dict = {'arg_formatters':'model_function',
                          'model_function_formatter':'model_function'}

        if parametric_model_class is HistParametricModel:
            _override_dict['model_density_function_name'] = 'model_function'
            _override_dict['latex_model_density_function_name'] = 'model_function'
            _override_dict['x_name'] = 'model_function'
            _override_dict['latex_x_name'] = 'model_function'
            _override_dict['expression_string'] = 'model_function'
            _override_dict['latex_expression_string'] = 'model_function'
        elif parametric_model_class is IndexedParametricModel:
            _override_dict['model_function_name'] = 'model_function'
            _override_dict['latex_model_function_name'] = 'model_function'
            _override_dict['index_name'] = 'model_function'
            _override_dict['latex_index_name'] = 'model_function'
            _override_dict['expression_string'] = 'model_function'
            _override_dict['latex_expression_string'] = 'model_function'
        elif parametric_model_class is XYParametricModel:
            _override_dict['model_function_name'] = 'model_function'
            _override_dict['latex_model_function_name'] = 'model_function'
            _override_dict['x_name'] = 'model_function'
            _override_dict['latex_x_name'] = 'model_function'
            _override_dict['expression_string'] = 'model_function'
            _override_dict['latex_expression_string'] = 'model_function'
        elif parametric_model_class is XYMultiParametricModel:
            for _i in range(10):
                _override_dict['model_function_name_%s' % _i] = 'model_function'
                _override_dict['latex_model_function_name_%s' % _i] = 'model_function'
                _override_dict['x_name_%s' % _i] = 'model_function'
                _override_dict['latex_x_name_%s' % _i] = 'model_function'
                _override_dict['expression_string_%s' % _i] = 'model_function'
                _override_dict['latex_expression_string_%s' % _i] = 'model_function'
            _override_dict['x_name'] = 'model_function'
            _override_dict['latex_x_name'] = 'model_function'
        else:
            raise YamlReaderException("Unknown parametric model type")
        return _override_dict
    
    @classmethod
    def _get_required_keywords(cls, yaml_doc, parametric_model_class):
        if parametric_model_class is HistParametricModel:
            return []
        elif parametric_model_class is IndexedParametricModel:
            return ['model_function']
        elif parametric_model_class is XYParametricModel:
            return ['x_data']
        elif parametric_model_class is XYMultiParametricModel:
            _required_keywords = ['x_data_0']
            if 'model_function' not in list(yaml_doc.keys()):
                _required_keywords.append('model_function_0')
            return _required_keywords
    
    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine model function class from type
        _parametric_model_type = yaml_doc.pop('type')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_parametric_model_type, None)

        #_kwarg_list = kwargs that directly correspond to unchanging scalar yaml entries
        _kwarg_list = []         
        _constructor_kwargs = {}
        if _class is HistParametricModel:
            if 'bin_edges' in yaml_doc:
                _bin_edges = yaml_doc.pop('bin_edges')
                # bin_edges overrides n_bins and bin_range
                yaml_doc.pop('n_bins', None)
                yaml_doc.pop('bin_range', None)
                _constructor_kwargs['bin_edges'] = _bin_edges
                _constructor_kwargs['n_bins'] = len(_bin_edges) - 1
                _constructor_kwargs['bin_range'] = [_bin_edges[0], _bin_edges[-1]]
                
            elif 'n_bins' in yaml_doc and 'bin_range' in yaml_doc:
                _kwarg_list.append('n_bins')
                _kwarg_list.append('bin_range')
            else:
                raise YamlReaderException("When reading in a histogram parametric model either "
                                          "bin_edges or n_bins and bin_range have to be specified!")
            #TODO implement parsing
            _kwarg_list.append('model_density_func_antiderivative')
        elif _class is IndexedParametricModel:
            _kwarg_list.append('shape_like')
        elif _class is XYParametricModel:
            _kwarg_list.append('x_data')
            yaml_doc.pop('y_data', None) # remove y_data from dict
        elif _class is XYMultiParametricModel:
            _x_data = []
            _data_indices = [0] #indices at which a new dataset begins or ends
            _i = 0 #xy dataset index
            _x_data_i = yaml_doc.pop('x_data_%s' % _i, None)
            # y data is written by ParametricModelYamlWriter for human understanding
            # it is popped here to remove it from yaml_doc but it is not being used
            _y_data_i = yaml_doc.pop('y_data_%s' % _i, None)
            while _x_data_i is not None:
                _x_data += _x_data_i
                _i += 1
                _data_indices.append(len(_x_data))
                _x_data_i = yaml_doc.pop('x_data_%s' % _i, None)
                _y_data_i = yaml_doc.pop('y_data_%s' % _i, None)
            _constructor_kwargs['x_data'] = _x_data
        else:
            raise YamlReaderException("Unkonwn parametric model type")

        _model_func = None
        if _class is HistParametricModel:
            _model_func_entry = yaml_doc.pop('model_density_function', None)
            if _model_func_entry:
                _model_func = ModelFunctionYamlReader._make_object(_model_func_entry, default_type=_parametric_model_type)
                _constructor_kwargs['model_density_func'] = _model_func
        elif _class in (IndexedParametricModel, XYParametricModel):
            _model_func_entry = yaml_doc.pop('model_function', None)
            if _model_func_entry:
                _model_func = ModelFunctionYamlReader._make_object(_model_func_entry, default_type=_parametric_model_type)
                _constructor_kwargs['model_func'] = _model_func
        elif _class is XYMultiParametricModel:
            _model_func_entry = yaml_doc.pop('model_function', None)
            if _model_func_entry:
                _model_func = ModelFunctionYamlReader._make_object(_model_func_entry, default_type=_parametric_model_type)
            else:
                _model_function_list = []
                for _i in range(len(_data_indices) - 1):
                    _model_function_i = yaml_doc.pop('model_function_%s' % _i, None)
                    if not _model_function_i:
                        raise YamlReaderException(
                            'There is no corresponding model_function_%s entry for x_data_%s' % (_i, _i))
                    if isinstance(_model_function_i, str):
                        _function_library_entry = function_library.STRING_TO_FUNCTION.get(_model_function_i, None)
                        if _function_library_entry:
                            _model_function_list.append(_function_library_entry)
                        else:
                            _model_function_list.append(ModelFunctionYamlReader._parse_model_function(_model_function_i))
                    else:
                        _model_function_list.append(_model_function_i)
                _model_func = XYMultiModelFunction(
                    model_function_list=_model_function_list,
                    data_indices=_data_indices
                )
            _constructor_kwargs['model_func'] = _model_func
                
        else:
            raise YamlReaderException('Unknown model type: %s' % _parametric_model_type)

        if _model_func:
            # if model parameters are given, apply those to the model function
            # if not use model function defaults            
            _given_parameters = yaml_doc.pop('model_parameters', None)
            if _given_parameters:
                _model_func.defaults = _given_parameters
                _constructor_kwargs['model_parameters'] = _given_parameters
            else:
                _constructor_kwargs['model_parameters'] = _model_func.defaults
        
        
        _constructor_kwargs.update({key: yaml_doc.pop(key, None) for key in _kwarg_list})
        _parametric_model_object = _class(**_constructor_kwargs)
        
        # -- process error sources
        if _class in (XYParametricModel, XYMultiParametricModel):
            _xerrs = yaml_doc.pop('x_errors', [])
            _yerrs = yaml_doc.pop('y_errors', [])
            _errs = _xerrs + _yerrs
            _axes = [0] * len(_xerrs) + [1] * len(_yerrs)  # 0 for 'x', 1 for 'y'
        else:
            _errs = yaml_doc.pop('errors', [])
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
        
        return _parametric_model_object, yaml_doc
    
# register the above classes in the module-level dictionary
ModelFunctionYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
ParametricModelYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ParametricModelYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
