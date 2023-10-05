import inspect

import numpy as np

from ....fit import CustomFit, HistFit, IndexedFit, UnbinnedFit, XYFit
from ....tools import get_compact_representation
from ..._base.cost import STRING_TO_COST_FUNCTION
from ..._base.format import ParameterFormatter
from ...unbinned.cost import STRING_TO_COST_FUNCTION as STRING_TO_COST_FUNCTION_UNBINNED
from ...util import to_numpy_arrays, to_python_types
from ...xy.cost import STRING_TO_COST_FUNCTION as STRING_TO_COST_FUNCTION_XY
from .. import _AVAILABLE_REPRESENTATIONS
from .._yaml_base import YamlReaderMixin, YamlWriterMixin
from ..constraint.yaml_drepr import ConstraintYamlReader, ConstraintYamlWriter
from ..container.yaml_drepr import DataContainerYamlReader, DataContainerYamlWriter
from ..model.yaml_drepr import (
    ParametricModelYamlReader,
    ParametricModelYamlWriter,
    _parse_function,
    _process_function_code_for_dump,
)
from ._base import FitDReprBase

__all__ = ["FitYamlWriter", "FitYamlReader"]


class FitYamlWriter(YamlWriterMixin, FitDReprBase):
    def __init__(self, fit, output_io_handle):
        super(FitYamlWriter, self).__init__(output_io_handle=output_io_handle, fit=fit)

    def _get_preface_comment(self):
        _preface_comment = super(FitYamlWriter, self)._get_preface_comment()
        _did_fit = self._kafe_object.did_fit
        if not _did_fit:
            _preface_comment += "\n# WARNING: No fit has been performed as of yet. " "Did you forget to run fit.do_fit()?\n"
        _preface_comment += "\n"
        if self._kafe_object.model_count == 1:
            _preface_comment += "# Model function: %s\n" % self._kafe_object._model_function.formatter.get_formatted(
                format_as_latex=False, with_expression=True, with_par_values=False
            )
        else:
            for _i in range(self._kafe_object.model_count):
                _preface_comment += "# Model function %s: %s\n" % (
                    _i,
                    self._kafe_object._model_function.formatter.get_formatted(
                        model_index=_i,
                        format_as_latex=False,
                        with_expression=True,
                        with_par_values=False,
                    ),
                )

        if _did_fit:
            _gof = self._kafe_object.goodness_of_fit
            _gof_name = "chi2" if self._kafe_object._cost_function.is_chi2 else "GoF"
            _cost = self._kafe_object.cost_function_value
            _ndf = self._kafe_object.ndf
            if _gof is None:
                _round_cost_sig = max(2, int(-np.floor(np.log(np.abs(_cost)) / np.log(10))) + 1)
                _rounded_cost = round(_cost, _round_cost_sig)
                _preface_comment += "# Cost: %s\n" % _rounded_cost
            else:
                _preface_comment += "# %s: %s\n" % (_gof_name, _gof)
                _round_gof_per_ndf_sig = max(2, int(-np.floor(np.log(np.abs(_gof) / _ndf) / np.log(10))) + 1)
            if _ndf is not None:
                _preface_comment += "# ndf: %s\n" % _ndf
            if _gof is not None:
                _preface_comment += "# %s/ndf: %s\n\n" % (
                    _gof_name,
                    round(_gof / _ndf, _round_gof_per_ndf_sig),
                )

            # If asymmetric parameters errors were not calculated, check the loaded result dict
            _asymmetric_parameter_errors = self._kafe_object._fitter.asymmetric_fit_parameter_errors_if_calculated
            if _asymmetric_parameter_errors is None and self._kafe_object._loaded_result_dict is not None:
                _asymmetric_parameter_errors = self._kafe_object._loaded_result_dict["asymmetric_parameter_errors"]

            _preface_comment += get_compact_representation(
                parameter_names=self._kafe_object.parameter_names,
                parameter_values=self._kafe_object.parameter_values,
                parameter_errors=self._kafe_object.parameter_errors,
                parameter_cor_mat=self._kafe_object.parameter_cor_mat,
                asymmetric_parameter_errors=_asymmetric_parameter_errors,
            )
        _preface_comment += "\n"
        return _preface_comment

    @classmethod
    def _make_representation(cls, fit):
        _yaml_doc = dict()

        # -- determine fit type
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(fit.__class__, None)
        if _type is None:
            raise TypeError("Fit type unknown or not supported: %s" % fit.__class__)
        _yaml_doc["type"] = _type

        if _type != "custom":
            _yaml_doc["dataset"] = DataContainerYamlWriter._make_representation(fit.data_container)
            _yaml_doc["parametric_model"] = ParametricModelYamlWriter._make_representation(fit._param_model)
        else:
            _par_formatter_dict = {_par_formatter.arg_name: _par_formatter.name for _par_formatter in fit._parameter_formatters}
            if _par_formatter_dict:
                _yaml_doc["parameter_formatters"] = _par_formatter_dict

        _cost_function_identifier = fit._cost_function.kafe2go_identifier
        if _cost_function_identifier is not None:
            _yaml_doc["cost_function"] = _cost_function_identifier
        else:
            _yaml_doc["cost_function"] = _process_function_code_for_dump(inspect.getsource(fit._cost_function.func))

        _yaml_doc["minimizer"] = fit._minimizer
        _yaml_doc["minimizer_kwargs"] = fit._minimizer_kwargs

        _yaml_doc["parameter_constraints"] = [
            ConstraintYamlWriter._make_representation(_parameter_constraint) for _parameter_constraint in fit.parameter_constraints
        ]
        _yaml_doc["fixed_parameters"] = fit._fitter.fixed_parameters
        _yaml_doc["limited_parameters"] = {_par_name: list(_par_limits) for _par_name, _par_limits in fit._fitter.limited_parameters.items()}

        _yaml_doc["fit_results"] = to_python_types(fit.get_result_dict())
        return _yaml_doc


class FitYamlReader(YamlReaderMixin, FitDReprBase):
    def __init__(self, input_io_handle):
        super(FitYamlReader, self).__init__(input_io_handle=input_io_handle, fit=None)

    @classmethod
    def _get_required_keywords(cls, yaml_doc, fit_class):
        if fit_class is CustomFit:
            return ["cost_function"]
        if fit_class in (HistFit, XYFit):
            return ["dataset"]
        return ["dataset", "parametric_model"]

    @classmethod
    def _get_subspace_override_dict(cls, fit_class):
        _override_dict = {
            "model_parameters": "parametric_model",
            "arg_formatters": "parametric_model",
            "model_function_formatter": "parametric_model",
            "expression_string": "parametric_model",
            "latex_expression_string": "parametric_model",
        }
        if fit_class is CustomFit:
            pass
        elif fit_class is HistFit:
            _override_dict["n_bins"] = ["dataset", "parametric_model"]
            _override_dict["bin_range"] = ["dataset", "parametric_model"]
            _override_dict["bin_edges"] = ["dataset", "parametric_model"]
            _override_dict["raw_data"] = "dataset"
            _override_dict["errors"] = "dataset"
            _override_dict["model_density_function"] = "parametric_model"
            _override_dict["model_density_function_name"] = "parametric_model"
            _override_dict["latex_model_density_function_name"] = "parametric_model"
        elif fit_class is IndexedFit:
            _override_dict["data"] = "dataset"
            _override_dict["errors"] = "dataset"
            _override_dict["model_function"] = "parametric_model"
            _override_dict["model_function_name"] = "parametric_model"
            _override_dict["latex_model_function_name"] = "parametric_model"
            _override_dict["index_name"] = "parametric_model"
            _override_dict["latex_index_name"] = "parametric_model"
        elif fit_class is UnbinnedFit:
            _override_dict["data"] = ["dataset", "parametric_model"]
            _override_dict["model_function"] = "parametric_model"
            _override_dict["model_function_name"] = "parametric_model"
            _override_dict["latex_model_function_name"] = "parametric_model"
        elif fit_class is XYFit:
            _override_dict["x_data"] = ["dataset", "parametric_model"]
            _override_dict["y_data"] = "dataset"
            _override_dict["x_errors"] = "dataset"
            _override_dict["y_errors"] = "dataset"
            _override_dict["model_function"] = "parametric_model"
            _override_dict["model_function_name"] = "parametric_model"
            _override_dict["latex_model_function_name"] = "parametric_model"
        else:
            raise TypeError("Unknown fit type")
        # override labels for every fit type
        _override_dict["label"] = "dataset"
        _override_dict["model_label"] = "parametric_model"
        _override_dict["x_label"] = "dataset"
        _override_dict["y_label"] = "dataset"
        return _override_dict

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine fit class from type
        _fit_type = yaml_doc.pop("type")
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_fit_type, None)

        if _fit_type != "custom":
            _data = DataContainerYamlReader._make_object(yaml_doc.pop("dataset"), default_type=_fit_type)
        _parametric_model_entry = yaml_doc.pop("parametric_model", None)
        if _parametric_model_entry:
            _read_parametric_model = ParametricModelYamlReader._make_object(_parametric_model_entry, default_type=_fit_type, dataset=_data)
            _read_model_function = _read_parametric_model._model_function_object
        else:
            _read_parametric_model = None
            _read_model_function = None

        _cost_function = yaml_doc.pop("cost_function", None)
        if _cost_function is not None:
            if _fit_type == "xy":
                _lookup_dict = STRING_TO_COST_FUNCTION_XY
            elif _fit_type == "unbinned":
                _lookup_dict = STRING_TO_COST_FUNCTION_UNBINNED
            else:
                _lookup_dict = STRING_TO_COST_FUNCTION
            if _cost_function not in _lookup_dict:
                _cost_function = _parse_function(_cost_function)

        _minimizer = yaml_doc.pop("minimizer", None)
        _minimizer_kwargs = yaml_doc.pop("minimizer_kwargs", None)
        # change fit kwargs for different fit types if necessary
        _fit_kwargs = dict(minimizer=_minimizer, minimizer_kwargs=_minimizer_kwargs)
        if _cost_function is not None:
            _fit_kwargs["cost_function"] = _cost_function
        if _fit_type != "custom":
            _fit_object = _class(_data, _read_model_function, **_fit_kwargs)
        else:
            _fit_object = _class(**_fit_kwargs)
            _par_formatters = yaml_doc.pop("parameter_formatters", None)
            if _par_formatters is not None:
                _fit_object._parameter_formatters = [
                    ParameterFormatter(arg_name=_arg_name, name=_name) for (_arg_name, _name) in _par_formatters.items()
                ]
            _fit_object._update_parameter_formatters()

        if _read_parametric_model is not None:
            _fit_object._param_model = _read_parametric_model

        _constraint_yaml_list = yaml_doc.pop("parameter_constraints", None)
        if isinstance(_constraint_yaml_list, dict):
            _constraint_yaml_list = [dict(name=param, **_constraint_yaml) for param, _constraint_yaml in _constraint_yaml_list.items()]
        if isinstance(_constraint_yaml_list, list):
            _fit_object._fit_param_constraints = [
                ConstraintYamlReader._make_object(
                    _constraint_yaml,
                    default_type="simple",
                    parameter_names=_fit_object.parameter_names,
                )
                for _constraint_yaml in _constraint_yaml_list
            ]

        _fixed_par_list = yaml_doc.pop("fixed_parameters", None)
        if _fixed_par_list is not None:
            for _par, _val in _fixed_par_list.items():
                _fit_object.fix_parameter(_par, _val)

        _limited_par_list = yaml_doc.pop("limited_parameters", None)
        if _limited_par_list is not None:
            for _par, _limits in _limited_par_list.items():
                _low, _high = _limits
                _fit_object.limit_parameter(_par, _low, _high)

        _fit_results = yaml_doc.pop("fit_results", None)
        _fit_object._loaded_result_dict = to_numpy_arrays(_fit_results)
        return _fit_object, yaml_doc


# register the above classes in the module-level dictionary
FitYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
FitYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
