from .._base import DReprError
from kafe.fit.representation._yaml_base import  YamlWriterMixin, YamlReaderMixin,\
    YamlReaderException
from ._base import FitDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from kafe.fit.histogram.fit import HistFit
from kafe.fit.representation.container.yaml_drepr import DataContainerYamlReader,\
    DataContainerYamlWriter
from kafe.fit.indexed.fit import IndexedFit
from kafe.fit.xy.fit import XYFit
from kafe.fit.representation.model.yaml_drepr import ParametricModelYamlReader,\
    ParametricModelYamlWriter
from kafe.fit.xy_multi.fit import XYMultiFit

__all__ = ['FitYamlWriter', 'FitYamlReader']

class FitYamlWriter(YamlWriterMixin, FitDReprBase):

    def __init__(self, fit, output_io_handle):
        super(FitYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            fit=fit)
    
    @classmethod
    def _make_representation(cls, fit):
        _yaml_doc = dict()

        # -- determine model function type
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(fit.__class__, None)
        if _type is None:
            raise DReprError("Fit type unknown or not supported: %s" % fit.__class__)
        _yaml_doc['type'] = _type
        
        _yaml_doc['dataset'] = DataContainerYamlWriter._make_representation(fit._data_container)
        _yaml_doc['parametric_model'] = ParametricModelYamlWriter._make_representation(fit._param_model)
        
        #TODO cost function
        
        _yaml_doc['minimizer'] = fit._minimizer
        _yaml_doc['minimizer_kwargs'] = fit._minimizer_kwargs
        
        return _yaml_doc
    
class FitYamlReader(YamlReaderMixin, FitDReprBase):
    
    def __init__(self, input_io_handle):
        super(FitYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            fit=None)

    @classmethod
    def _get_required_keywords(cls, yaml_doc, fit_class):
        if fit_class in (HistFit, XYFit):
            return ['dataset']
        else:
            return ['dataset', 'parametric_model']
    
    @classmethod
    def _get_subspace_override_dict(cls, fit_class):
        _override_dict = {'model_parameters':'parametric_model',
                          'arg_formatters':'parametric_model',
                          'model_function_formatter':'parametric_model'}

        if fit_class is HistFit:
            _override_dict['n_bins'] = ['dataset', 'parametric_model']
            _override_dict['bin_range'] = ['dataset', 'parametric_model']
            _override_dict['bin_edges'] = ['dataset', 'parametric_model']
            _override_dict['raw_data'] = 'dataset'
            _override_dict['errors'] = 'dataset'
            _override_dict['model_density_function'] = 'parametric_model'
            _override_dict['model_density_function_name'] = 'parametric_model'
            _override_dict['latex_model_density_function_name'] = 'parametric_model'
            _override_dict['x_name'] = 'parametric_model'
            _override_dict['latex_x_name'] = 'parametric_model'
            _override_dict['expression_string'] = 'parametric_model'
            _override_dict['latex_expression_string'] = 'parametric_model'
        elif fit_class is IndexedFit:
            _override_dict['data'] = 'dataset'
            _override_dict['errors'] = 'dataset'
            _override_dict['model_function'] = 'parametric_model'
            _override_dict['model_function_name'] = 'parametric_model'
            _override_dict['latex_model_function_name'] = 'parametric_model'
            _override_dict['index_name'] = 'parametric_model'
            _override_dict['latex_index_name'] = 'parametric_model'
            _override_dict['expression_string'] = 'parametric_model'
            _override_dict['latex_expression_string'] = 'parametric_model'
        elif fit_class is XYFit:
            _override_dict['x_data'] = 'dataset'
            _override_dict['y_data'] = 'dataset'
            _override_dict['x_errors'] = 'dataset'
            _override_dict['y_errors'] = 'dataset'
            _override_dict['model_function'] = 'parametric_model'
            _override_dict['model_function_name'] = 'parametric_model'
            _override_dict['latex_model_function_name'] = 'parametric_model'
            _override_dict['x_name'] = 'parametric_model'
            _override_dict['latex_x_name'] = 'parametric_model'
            _override_dict['expression_string'] = 'parametric_model'
            _override_dict['latex_expression_string'] = 'parametric_model'
        elif fit_class is XYMultiFit:
            _override_dict['x_errors'] = 'dataset'
            _override_dict['y_errors'] = 'dataset'
            for _i in range(10): #TODO config
                _override_dict['x_data_%s' % _i] = 'dataset'
                _override_dict['y_data_%s' % _i] = 'dataset'
                _override_dict['model_function_%s' % _i] = 'parametric_model'
                _override_dict['model_function_name_%s' % _i] = 'parametric_model'
                _override_dict['latex_model_function_name_%s' % _i] = 'parametric_model'
                _override_dict['x_name_%s' % _i] = 'parametric_model'
                _override_dict['latex_x_name_%s' % _i] = 'parametric_model'
                _override_dict['expression_string_%s' % _i] = 'parametric_model'
                _override_dict['latex_expression_string_%s' % _i] = 'parametric_model'
            _override_dict['x_name'] = 'parametric_model'
            _override_dict['latex_x_name'] = 'parametric_model'
        else:
            raise YamlReaderException("Unknown fit type")
        return _override_dict
    
    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine model function class from type
        _fit_type = yaml_doc.pop('type')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_fit_type, None)
        
        _data = DataContainerYamlReader._make_object(yaml_doc.pop('dataset'), default_type=_fit_type)
        _parametric_model_entry = yaml_doc.pop('parametric_model', None)
        if _parametric_model_entry:
            _read_parametric_model = ParametricModelYamlReader._make_object(_parametric_model_entry, default_type=_fit_type)
            _read_model_function = _read_parametric_model._model_function_object
        else:
            _read_parametric_model = None
            _read_model_function = None
        #TODO cost function
        _minimizer = yaml_doc.pop('minimizer', None)
        _minimizer_kwargs = yaml_doc.pop('minimizer_kwargs', None)
        if _class is HistFit:
            _fit_object = HistFit(
                data=_data,
                model_density_function=_read_model_function,
                model_density_antiderivative=None,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs
            )
        elif _class is IndexedFit:
            _fit_object = IndexedFit(
                data=_data,
                model_function=_read_model_function,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs                
            )
        elif _class is XYFit:
            _fit_object = XYFit(
                xy_data=_data,
                model_function=_read_model_function,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs
            )
        elif _class is XYMultiFit:
            _fit_object = XYMultiFit(
                xy_data=_data,
                model_function=_read_model_function,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs
            )
        if _read_parametric_model:
            _fit_object._param_model = _read_parametric_model
        return _fit_object, yaml_doc
    
# register the above classes in the module-level dictionary
FitYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
FitYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
