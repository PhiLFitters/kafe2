import textwrap
import tokenize

import numpy as np
import six

from ._base import ModelFunctionDReprBase, ParametricModelDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from .._base import DReprError
from .._yaml_base import YamlReaderException, YamlReaderMixin, YamlWriterException, YamlWriterMixin
from ..container import DataContainerYamlReader, DataContainerYamlWriter
from ..format import ModelFunctionFormatterYamlReader, ModelFunctionFormatterYamlWriter
from ... import HistModelFunction, HistParametricModel, IndexedModelFunction, \
    IndexedParametricModel, UnbinnedParametricModel, XYParametricModel
from ..._base import ModelFunctionBase
from ....fit.util import function_library

__all__ = ['ModelFunctionYamlWriter', 'ModelFunctionYamlReader', 'ParametricModelYamlWriter',
           'ParametricModelYamlReader']
KNOWN_MODEL_FUNCTIONS = (ModelFunctionBase, HistModelFunction, IndexedModelFunction)
KNOWN_PARAMETRIC_MODELS = (XYParametricModel, HistParametricModel, IndexedParametricModel,
                           UnbinnedParametricModel)


def _parse_function(input_string):
    """converts a string of python code into a python function object"""
    _tokens = tokenize.generate_tokens(six.StringIO(input_string).readline)
    for _toknum, _tokval, _spos, _epos, _line_string in _tokens:
        if _tokval in ModelFunctionYamlReader.FORBIDDEN_TOKENS:
            raise DReprError("Encountered forbidden token '%s' in user-entered code on line '%s'."
                             % (_tokval, _line_string))

    if "__" in input_string:
        raise DReprError("Model function input must not contain '__'!")

    _imports = ""
    _imports += "import numpy as np\n"  # import numpy
    # import scipy if installed
    try:
        import scipy
        _imports += "import scipy\n"
    except ImportError:
        pass

    __locals_pointer = [None, None]  # TODO better solution?
    # save locals before function definition
    _exec_string = _imports + "__locals_pointer[0] = __locals().copy()\n" + input_string
    # save locals after function definition
    _exec_string = _exec_string + "\n__locals_pointer[1] = __locals().copy()"
    _restricted_builtins = __builtins__.copy()
    for _forbidden_token in ModelFunctionYamlReader.FORBIDDEN_TOKENS:
        _restricted_builtins.pop(_forbidden_token, None)
    _restricted_builtins['__import__'] = __import__
    exec(_exec_string, {"__builtins__": _restricted_builtins, "__locals": locals,
                        "__locals_pointer": __locals_pointer})
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


def _process_function_code_for_dump(source_code):
    source_code = textwrap.dedent(source_code)  # remove indentation
    source_code = source_code.replace("@staticmethod\n", "")  # remove @staticmethod decorator
    # TODO what about other decorators?
    return source_code


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
        if _type != 'base':  # 'base' is default, no need to store type
            _yaml_doc['type'] = _type  # store all other custom types
        _yaml_doc['model_function_formatter'] =\
            ModelFunctionFormatterYamlWriter._make_representation(model_function.formatter)

        _yaml_doc['python_code'] = _process_function_code_for_dump(model_function.source_code)

        return _yaml_doc


class ModelFunctionYamlReader(YamlReaderMixin, ModelFunctionDReprBase):
    FORBIDDEN_TOKENS = ['compile', 'eval', 'exec', 'execfile', 'file', 'global', 'globals',
                        'import', '__import__',
                        'input', ' locals', 'nonlocal', 'open', 'reload', 'self', 'super']

    def __init__(self, input_io_handle):
        super(ModelFunctionYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            model_function=None)

    @classmethod
    def _get_subspace_override_dict(cls, model_function_class):
        _override_dict = {'name': 'model_function_formatter',
                          'latex_name': 'model_function_formatter',
                          'arg_formatters': 'model_function_formatter',
                          'expression_string': 'model_function_formatter',
                          'latex_expression_string': 'model_function_formatter',
                          'signature': 'model_function_formatter'}
        if model_function_class is IndexedModelFunction:
            _override_dict['index_name'] = 'model_function_formatter'
            _override_dict['latex_index_name'] = 'model_function_formatter'
        return _override_dict

    @classmethod
    def _get_required_keywords(cls, yaml_doc, model_function_class):
        if issubclass(model_function_class, ModelFunctionBase):
            return ['python_code']
        raise YamlReaderException("Unknown model function class: %s" % model_function_class)

    @classmethod
    def _process_string(cls, string_representation, default_type):
        return dict(type=default_type, python_code=string_representation)

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        """Converts the given yaml dictionary to a :py:obj:``ModelFunctionBase`` or derived object.

        :param yaml_doc: the yaml doc to convert
        :type yaml_doc: dict"""
        # -- determine model function class from type
        _model_function_type = yaml_doc.pop('type', 'base')  # if no type is specified assume base
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_model_function_type)
        if _class not in KNOWN_MODEL_FUNCTIONS:
            raise YamlReaderException("Unknown model function class: %s" % _class)

        _raw_string = yaml_doc.pop("python_code")
        _function_library_entry = function_library.STRING_TO_FUNCTION.get(_raw_string, None)
        if _function_library_entry:
            _model_function_object = _class(_function_library_entry)
        else:
            _parsed_function = _parse_function(_raw_string)
            _model_function_object = _class(_parsed_function)
            _model_function_object._source_code = _raw_string

        # construct model function formatter if specified
        _model_function_formatter_yaml = yaml_doc.pop('model_function_formatter', None)
        if _model_function_formatter_yaml:
            _name = _model_function_object.name
            _signature = _model_function_object.signature.parameters
            _model_function_object._formatter = ModelFunctionFormatterYamlReader._make_object(
                _model_function_formatter_yaml, default_type=_model_function_type, name=_name,
                signature=_signature)

        return _model_function_object, yaml_doc


class ParametricModelYamlWriter(YamlWriterMixin, ParametricModelDReprBase):

    def __init__(self, parametric_model, output_io_handle):
        super(ParametricModelYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            parametric_model=parametric_model)

    @classmethod
    def _make_representation(cls, parametric_model):
        """Create a dictionary representation of a parametric model.

        :param parametric_model: The parametric model to convert.
        :type parametric_model: kafe2.fit.histogram.HistParametricModel | kafe2.fit.indexed.IndexedParametricModel |
                                kafe2.fit.xy.XYParametricModel
        """
        _yaml_doc = dict()

        # -- determine model function type
        _class = parametric_model.__class__
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Parametric model type unknown or not supported: %s" % _class)
        _yaml_doc['type'] = _type

        # -- write representation for model types
        if _class is HistParametricModel:
            _yaml_doc['n_bins'] = int(parametric_model.n_bins)
            _yaml_doc['bin_range'] = list(map(float, parametric_model.bin_range))
            _yaml_doc['model_density_function'] = ModelFunctionYamlWriter._make_representation(
                parametric_model._model_function_object)
            _yaml_doc['bin_edges'] = list(map(float, parametric_model.bin_edges))
            if isinstance(parametric_model.bin_evaluation, str):
                _yaml_doc['bin_evaluation'] = parametric_model.bin_evaluation_string
            else:
                _yaml_doc['bin_evaluation'] = _process_function_code_for_dump(
                    parametric_model.bin_evaluation_string)
        elif _class is IndexedParametricModel:
            _yaml_doc['shape_like'] = parametric_model.data.tolist()
            _yaml_doc['model_function'] = ModelFunctionYamlWriter._make_representation(
                parametric_model._model_function_object)
        elif _class is UnbinnedParametricModel:
            _yaml_doc['data'] = parametric_model.support.tolist()
            _yaml_doc['model_function'] = ModelFunctionYamlWriter._make_representation(
                parametric_model._model_function_object)
        elif _class is XYParametricModel:
            _yaml_doc['x_data'] = parametric_model.x.tolist()
            _yaml_doc['y_data'] = parametric_model.y.tolist()
            _yaml_doc['model_function'] = ModelFunctionYamlWriter._make_representation(
                parametric_model._model_function_object)
        else:
            raise YamlWriterException("Unkonwn parametric model type")

        # convert all numpy array entries to regular float, then convert to list, improves readability
        _yaml_doc['model_parameters'] = np.array(parametric_model.parameters, dtype=float).tolist()

        # write model label for all types
        _yaml_doc['model_label'] = parametric_model.label

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
        _override_dict = {'arg_formatters': 'model_function',
                          'model_function_formatter': 'model_function'}

        if parametric_model_class is HistParametricModel:
            _override_dict['model_density_function_name'] = 'model_function'
            _override_dict['latex_model_density_function_name'] = 'model_function'
            _override_dict['expression_string'] = 'model_function'
            _override_dict['latex_expression_string'] = 'model_function'
        elif parametric_model_class is IndexedParametricModel:
            _override_dict['model_function_name'] = 'model_function'
            _override_dict['latex_model_function_name'] = 'model_function'
            _override_dict['index_name'] = 'model_function'
            _override_dict['latex_index_name'] = 'model_function'
            _override_dict['expression_string'] = 'model_function'
            _override_dict['latex_expression_string'] = 'model_function'
        elif parametric_model_class is XYParametricModel or parametric_model_class is UnbinnedParametricModel:
            _override_dict['model_function_name'] = 'model_function'
            _override_dict['latex_model_function_name'] = 'model_function'
            _override_dict['expression_string'] = 'model_function'
            _override_dict['latex_expression_string'] = 'model_function'
        else:
            raise YamlReaderException("Unknown parametric model type")
        return _override_dict

    @classmethod
    def _get_required_keywords(cls, yaml_doc, parametric_model_class):
        if parametric_model_class is HistParametricModel:
            return []
        if parametric_model_class is UnbinnedParametricModel:
            return ['data']
        if parametric_model_class is IndexedParametricModel:
            return ['model_function']
        if parametric_model_class is XYParametricModel:
            return ['x_data']

    @classmethod
    def _get_ignored_if_none_keywords(cls):
        return ["model_density_func_antiderivative"]

    @classmethod
    def _modify_yaml_doc(cls, yaml_doc, kafe_object_class, dataset=None, **kwargs):
        if dataset:
            if kafe_object_class is HistParametricModel:
                if 'bin_edges' not in yaml_doc and ('n_bins' not in yaml_doc or 'bin_range' not
                                                    in yaml_doc):
                    yaml_doc['bin_edges'] = dataset.bin_edges
            elif kafe_object_class is IndexedParametricModel:
                if 'shape_like' not in yaml_doc:
                    yaml_doc['shape_like'] = dataset.data
            elif kafe_object_class is UnbinnedParametricModel:
                if 'data' not in yaml_doc:
                    yaml_doc['data'] = dataset.data
            elif kafe_object_class is XYParametricModel:
                if 'x_data' not in yaml_doc:
                    yaml_doc['x_data'] = dataset.x
        super(ParametricModelYamlReader, cls)._modify_yaml_doc(yaml_doc, kafe_object_class,
                                                               **kwargs)
        return yaml_doc

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine model function class from type
        _parametric_model_type = yaml_doc.pop('type')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_parametric_model_type, None)

        # _kwarg_list = kwargs that directly correspond to unchanging scalar yaml entries
        _kwarg_list = []
        _constructor_kwargs = {}
        _hist_model_bin_evaluation_source = None  # Set if an antiderivative function is read in.
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
            if 'bin_evaluation' in yaml_doc:
                _bin_evaluation = yaml_doc.pop('bin_evaluation')
                if 'def' in _bin_evaluation:
                    _constructor_kwargs['bin_evaluation'] = _parse_function(_bin_evaluation)
                    _hist_model_bin_evaluation_source = _bin_evaluation
                else:
                    _constructor_kwargs['bin_evaluation'] = _bin_evaluation
        elif _class is IndexedParametricModel:
            _kwarg_list.append('shape_like')
        elif _class is UnbinnedParametricModel:
            _kwarg_list.append('data')
        elif _class is XYParametricModel:
            _kwarg_list.append('x_data')
            yaml_doc.pop('y_data', None)  # remove y_data from dict
        else:
            raise YamlReaderException("Unknown parametric model type")

        _model_func = None
        if _class is HistParametricModel:
            _model_func_entry = yaml_doc.pop('model_density_function', None)
            if _model_func_entry:
                _model_func = ModelFunctionYamlReader._make_object(
                    _model_func_entry, default_type=_parametric_model_type)
                _constructor_kwargs['model_density_func'] = _model_func
        elif _class is UnbinnedParametricModel:
            _model_func_entry = yaml_doc.pop('model_function', None)
            if _model_func_entry:
                _model_func = ModelFunctionYamlReader._make_object(
                    _model_func_entry, default_type=_parametric_model_type)
                _constructor_kwargs['model_density_function'] = _model_func
        elif _class in (IndexedParametricModel, XYParametricModel):
            _model_func_entry = yaml_doc.pop('model_function', None)
            if _model_func_entry:
                _model_func = ModelFunctionYamlReader._make_object(
                    _model_func_entry, default_type=_parametric_model_type)
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
        if _hist_model_bin_evaluation_source is not None:
            _parametric_model_object._bin_evaluation_string = _hist_model_bin_evaluation_source

        # add the label for all types
        _parametric_model_object.label = yaml_doc.pop('model_label', None)

        # -- process error sources
        if _class is XYParametricModel:
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
                    _add_kwargs['err_val'] = _err['error_value']
                    _add_kwargs['correlation'] = _err['correlation_coefficient']
                elif _err_type == 'matrix':
                    _add_kwargs['err_matrix'] = _err['matrix']
                    _add_kwargs['matrix_type'] = _err['matrix_type']
                    # only mandatory for cor mats; check done later
                    _add_kwargs['err_val'] = _err.get('error_value', None)
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
            DataContainerYamlReader._add_error_to_container(_err_type, _parametric_model_object,
                                                            **_add_kwargs)

        return _parametric_model_object, yaml_doc


# register the above classes in the module-level dictionary
ModelFunctionYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ModelFunctionYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
ParametricModelYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ParametricModelYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
