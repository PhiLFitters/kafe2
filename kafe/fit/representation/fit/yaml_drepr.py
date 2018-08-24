import inspect
import numpy as np
import StringIO
import textwrap
import tokenize
import yaml

from .._base import DReprError, DReprWriterMixin, DReprReaderMixin
from ._base import FitDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from kafe.fit.histogram.fit import HistFit
from kafe.fit.representation.container.yaml_drepr import DataContainerYamlReader,\
    DataContainerYamlWriter
from kafe.fit.indexed.fit import IndexedFit
from kafe.fit.xy.fit import XYFit
from kafe.fit.representation.model.yaml_drepr import ParametricModelYamlReader,\
    ParametricModelYamlWriter

__all__ = ['FitYamlWriter', 'FitYamlReader']

class FitYamlWriter(DReprWriterMixin, FitDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'writer'

    def __init__(self, fit, output_io_handle):
        super(FitYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            fit=fit)
    
    @staticmethod
    def _make_representation(fit):
        _yaml = dict()

        # -- determine model function type
        _type = FitYamlWriter._FIT_CLASS_TO_TYPE_NAME.get(fit.__class__, None)
        if _type is None:
            raise DReprError("Model function unknown or not supported: %s" % fit.__class__)
        _yaml['type'] = _type
        
        _yaml['dataset'] = DataContainerYamlWriter._make_representation(fit._data_container)
        _yaml.update(ParametricModelYamlWriter._make_representation(fit._param_model))
        
        #TODO cost function
        
        _yaml['minimizer'] = fit._minimizer
        _yaml['minimizer_kwargs'] = fit._minimizer_kwargs
        
        return dict(fit=_yaml) # wrap inner yaml inside a 'fit' namespace
    
    def write(self):
        self._yaml = self._make_representation(self._fit)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml, _h, default_flow_style=False)

class FitYamlReader(DReprReaderMixin, FitDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'reader'
    FORBIDDEN_TOKENS = ['eval', 'exec', 'execfile', 'file', 'global', 'import', '__import__', 'input', 
                        'nonlocal', 'open', 'reload', 'self', 'super']
    
    def __init__(self, input_io_handle):
        super(FitYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            fit=None)

    @staticmethod
    def _make_object(yaml):
        _yaml = yaml["fit"]

        # -- determine model function class from type
        _fit_type = _yaml['type']
        _class = FitYamlReader._FIT_TYPE_NAME_TO_CLASS.get(_fit_type, None)
        if _class is None:
            raise DReprError("Model function type unknown or not supported: {}".format(_fit_type))
        
        _data = DataContainerYamlReader._make_object(_yaml['dataset'])
        _parametric_model = ParametricModelYamlReader._make_object(_yaml)
        #TODO cost function
        _minimizer = _yaml.get('minimizer', None)
        _minimizer_kwargs = _yaml.get('minimizer_kwargs', None)
        if _fit_type == 'histogram':
            _fit_object = HistFit(
                data=_data,
                model_density_function=_parametric_model._model_function_object,
                model_density_antiderivative=None,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs
            )
        elif _fit_type == 'indexed':
            _fit_object = IndexedFit(
                data=_data,
                model_function=_parametric_model._model_function_object,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs                
            )
        elif _fit_type == 'xy':
            _fit_object = XYFit(
                xy_data=_data,
                model_function=_parametric_model._model_function_object,
                minimizer=_minimizer,
                minimizer_kwargs=_minimizer_kwargs
            )
        _fit_object._param_model = _parametric_model
        return _fit_object
    
    def read(self):
        with self._ihandle as _h:
            self._yaml = yaml.load(_h)
        return self._make_object(self._yaml)

# register the above classes in the module-level dictionary
FitYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
FitYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
