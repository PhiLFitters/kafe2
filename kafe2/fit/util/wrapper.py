"""
The easiest way to use *kafe2* (as part of a *Python* program) is to use the wrapper functions
below.
These functions provide pre-configured pipelines for the most common use cases and do not require
the user to manually manage objects.
"""

__all__ = ["custom_fit", "hist_fit", "indexed_fit", "unbinned_fit", "xy_fit", "plot", "k2Fit"]

try:
    import typing  # help IDEs with type-hinting inside docstrings  # noqa: F401 (unused import)
except ImportError:
    pass

import os
import warnings
from copy import deepcopy
from glob import glob

import numpy as np

_fit_history = []


def _get_file_index():
    os.makedirs("results", exist_ok=True)
    _file_index = 0
    _globbed_files = glob(f"results/fit-{_file_index:04d}-*")
    while len(_globbed_files) > 0:
        _file_index += 1
        _globbed_files = glob(f"results/fit-{_file_index:04d}-*")
    return _file_index


def _fit_wrapper_generic(fit, p0, dp0, limits, fixed, constraints, report, profile, save):
    if p0 is not None:
        fit.set_all_parameter_values(p0)
    if dp0 is not None:
        fit.parameter_errors = dp0

    if limits is not None:
        if not isinstance(limits[0], (list, tuple)):
            limits = (limits,)
        for _limit in limits:
            fit.limit_parameter(*_limit)
    if fixed is not None:
        if not isinstance(fixed[0], (list, tuple)):
            fixed = (fixed,)
        for _fix in fixed:
            fit.fix_parameter(*_fix)
    if constraints is not None:
        if not isinstance(constraints[0], (list, tuple)):
            constraints = (constraints,)
        for _constraint in constraints:
            fit.add_parameter_constraint(*_constraint)

    _fit_results = fit.do_fit(asymmetric_parameter_errors=profile)
    _fit_results["fit"] = fit
    if report:
        fit.report(asymmetric_parameter_errors=profile)

    if save:
        _file_index = _get_file_index()
        fit.save_state(f"results/fit-{_file_index:04d}-results.yml")
        with open(f"results/fit-{_file_index:04d}-report.txt", "w", encoding="utf8") as _f:
            fit.report(_f, asymmetric_parameter_errors=profile)
    else:
        _file_index = None

    _fit_history.append(dict(fit=fit, profile=profile, file_index=_file_index))

    return _fit_results


def _add_error_to_fit_generic(fit, error, errors_rel_to_model, correlated=False, relative=False):
    if error is None:
        return
    error = np.asarray(error)
    _reference = "model" if errors_rel_to_model and relative else "data"
    if correlated:
        if error.ndim == 0:
            error = np.reshape(error, (1,))
        for _err in error:
            fit.add_error(_err, correlation=1.0, relative=relative, reference=_reference)
    else:
        if error.ndim == 2:
            fit.add_matrix_error(error, "cov", relative=relative, reference=_reference)
        else:
            fit.add_error(error, relative=relative, reference=_reference)


def custom_fit(
    cost_function,
    p0=None,
    dp0=None,
    limits=None,
    fixed=None,
    constraints=None,
    report=False,
    profile=True,
    save=True,
):
    """
    Built-in function for directly minimizing a cost function without any explicit model, data, or
    errors.

    :param cost_function: The cost function to be minimized as a native Python function.
    :type cost_function: typing.Callable
    :param p0: the initial parameter values for the fit.
    :type p0: typing.Sequence[float]
    :param dp0: the initial parameter step size for the fit.
    :type dp0: typing.Sequence[float]
    :param limits: limits to be applied to the model parameter. The expected format for each limit
        is an iterable consisting of the parameter name, the lower bound, and then the upper bound.
        An iterable of limits can be passed to limit multiple parameters.
    :type limits: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param fixed: Model parameter to be fixed. The expected format for each parameter is the
        parameter name followed by an optional value to which the parameter should be set prior to
        fixing. An iterable of (name, value) tuples can be passed to fix multiple parameters.
    :type fixed: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param constraints: constraints to be applied to the model parameter. The expected format for
        each constraint is an iterable consisting of the parameter name, the parameter mean, and
        then the parameter uncertainty. An iterable of constraints can be passed to limit multiple
        parameters.
    :type constraints: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param report: whether a report of the data and fit results should be printed to the console.
    :type report: bool
    :param profile: whether the profile likelihood method should be used for asymmetric parameter
        errors and profile/contour plots.
    :type profile: bool
    :param save: whether the fit results should be saved to disk under `results`.
    :type save: bool
    :return: the fit results.
    :rtype: dict
    """
    from kafe2.fit.custom.fit import CustomFit

    _fit = CustomFit(cost_function)
    return _fit_wrapper_generic(_fit, p0, dp0, limits, fixed, constraints, report, profile, save)


def hist_fit(
    model_function=None,
    data=None,
    n_bins=None,
    bin_range=None,
    bin_edges=None,
    p0=None,
    dp0=None,
    error=None,
    error_rel=None,
    error_cor=None,
    error_cor_rel=None,
    errors_rel_to_model=True,
    density=True,
    gauss_approximation=None,
    limits=None,
    fixed=None,
    constraints=None,
    report=False,
    profile=True,
    save=True,
):
    """
    Built-in function for fitting a (probability density) function to one-dimensional data by
    binning the data. The uncertainty on the bins is assumed to follow a Poisson distribution. If
    any errors are specified then the Poisson distribution is instead approximated by a Gaussian
    distribution.

    :param model_function: The model function as a native Python function where the first
        argument denotes the independent *x* variable. Alternatively an already defined
        :py:class:`~kafe2.fit.histogram.model.HistModelFunction` object. Defaults to a normal
        distribution.
    :type model_function: typing.Callable
    :param data: the data for the fit. Can be either raw data, the result of `np.histogram`, or a
        :py:class:`kafe2.fit.hist.container.HistContainer` object.
    :type data: typing.Sequence[float] or :py:class:`kafe2.fit.hist.container.HistContainer`
    :param n_bins: how many bins raw data should be split into.
    :type n_bins: int
    :param bin_range: the lower and upper bound for the bins specified by n_bins.
    :type bin_range: typing.Sequence[float] of length 2
    :param bin_edges: explicit bin edges for raw data. If ``None``, each bin will have the same
        width.
    :type bin_edges: typing.Sequence[float]
    :param p0: the initial parameter values for the fit.
    :type p0: typing.Sequence[float]
    :param dp0: the initial parameter step size for the fit.
    :type dp0: typing.Sequence[float]
    :param error: uncorrelated absolute error on the bin heights.
    :type error: float or typing.Sequence[float]
    :param error_rel: uncorrelated relative error on the bin heights.
    :type error_rel: float or typing.Sequence[float]
    :param error_cor: correlated absolute error on the bin heights.
    :type error_cor: float or typing.Sequence[float]
    :param error_cor_rel: correlated relative error on the bin heights.
    :type error_cor_rel: float or typing.Sequence[float]
    :param errors_rel_to_model: whether the relative *y* errors should be relative to the model.
        Otherwise they are relative to the data.
    :type errors_rel_to_model: bool
    :param density: whether the model is a probability density function and the data should be
        normalized to match it.
    :type density: bool
    :param limits: limits to be applied to the model parameter. The expected format for each limit
        is an iterable consisting of the parameter name, the lower bound, and then the upper bound.
        An iterable of limits can be passed to limit multiple parameters.
    :type limits: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param fixed: Model parameter to be fixed. The expected format for each parameter is the
        parameter name followed by an optional value to which the parameter should be set prior to
        fixing. An iterable of (name, value) tuples can be passed to fix multiple parameters.
    :type fixed: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param constraints: constraints to be applied to the model parameter. The expected format for
        each constraint is an iterable consisting of the parameter name, the parameter mean, and
        then the parameter uncertainty. An iterable of constraints can be passed to limit multiple
        parameters.
    :type constraints: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param report: whether a report of the data and fit results should be printed to the console.
    :type report: bool
    :param profile: whether the profile likelihood method should be used for asymmetric parameter
        errors and profile/contour plots.
    :type profile: bool
    :param save: whether the fit results should be saved to disk under `results`.
    :type save: bool
    :return: the fit results.
    :rtype: dict
    """
    from kafe2.fit.histogram.container import HistContainer
    from kafe2.fit.histogram.fit import HistFit

    data = HistContainer(n_bins, bin_range, bin_edges, data)

    if gauss_approximation is None:
        gauss_approximation = error is not None or error_rel is not None or error_cor is not None or error_cor_rel is not None
    _cost_function = "gauss_approximation" if gauss_approximation else "poisson"

    if model_function is None:
        _fit = HistFit(data, cost_function=_cost_function, density=density)
    else:
        _fit = HistFit(data, model_function, cost_function=_cost_function, density=density)

    _add_error_to_fit_generic(_fit, error, errors_rel_to_model)
    _add_error_to_fit_generic(_fit, error_cor, errors_rel_to_model, correlated=True)
    _add_error_to_fit_generic(_fit, error_rel, errors_rel_to_model, relative=True)
    _add_error_to_fit_generic(_fit, error_cor_rel, errors_rel_to_model, correlated=True, relative=True)

    return _fit_wrapper_generic(_fit, p0, dp0, limits, fixed, constraints, report, profile, save)


def indexed_fit(
    model_function=None,
    data=None,
    p0=None,
    dp0=None,
    error=None,
    error_rel=None,
    error_cor=None,
    error_cor_rel=None,
    errors_rel_to_model=True,
    limits=None,
    fixed=None,
    constraints=None,
    report=False,
    profile=True,
    save=True,
):
    from kafe2.fit.indexed import IndexedFit

    _fit = IndexedFit(data, model_function)

    _add_error_to_fit_generic(_fit, error, errors_rel_to_model)
    _add_error_to_fit_generic(_fit, error_cor, errors_rel_to_model, correlated=True)
    _add_error_to_fit_generic(_fit, error_rel, errors_rel_to_model, relative=True)
    _add_error_to_fit_generic(_fit, error_cor_rel, errors_rel_to_model, correlated=True, relative=True)

    return _fit_wrapper_generic(_fit, p0, dp0, limits, fixed, constraints, report, profile, save)


def unbinned_fit(
    model_function=None,
    data=None,
    p0=None,
    dp0=None,
    limits=None,
    fixed=None,
    constraints=None,
    report=False,
    profile=True,
    save=True,
):
    """
    Built-in function for directly fitting a probability density function to one-dimensional data
    without binning the data.

    :param model_function: The model function as a native Python function where the first
        argument denotes the independent *x* variable. Alternatively an already defined
        :py:class:`~kafe2.fit._base.model.ModelFunctionBase` object. Defaults to a straight line.
    :type model_function: typing.Callable
    :param data: the data values for the fit. Must be one-dimensional.
    :type data: typing.Sequence[float]
    :param p0: the initial parameter values for the fit.
    :type p0: typing.Sequence[float]
    :param dp0: the initial parameter step size for the fit.
    :type dp0: typing.Sequence[float]
    :param limits: limits to be applied to the model parameter. The expected format for each limit
        is an iterable consisting of the parameter name, the lower bound, and then the upper bound.
        An iterable of limits can be passed to limit multiple parameters.
    :type limits: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param fixed: Model parameter to be fixed. The expected format for each parameter is the
        parameter name followed by an optional value to which the parameter should be set prior to
        fixing. An iterable of (name, value) tuples can be passed to fix multiple parameters.
    :type fixed: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param constraints: constraints to be applied to the model parameter. The expected format for
        each constraint is an iterable consisting of the parameter name, the parameter mean, and
        then the parameter uncertainty. An iterable of constraints can be passed to limit multiple
        parameters.
    :type constraints: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param report: whether a report of the data and fit results should be printed to the console.
    :type report: bool
    :param profile: whether the profile likelihood method should be used for asymmetric parameter
        errors and profile/contour plots.
    :type profile: bool
    :param save: whether the fit results should be saved to disk under `results`.
    :type save: bool
    :return: the fit results.
    :rtype: dict
    """
    from kafe2.fit.unbinned import UnbinnedFit

    _fit = UnbinnedFit(data, model_function)

    return _fit_wrapper_generic(_fit, p0, dp0, limits, fixed, constraints, report, profile, save)


def xy_fit(
    model_function=None,
    x_data=None,
    y_data=None,
    p0=None,
    dp0=None,
    x_error=None,
    y_error=None,
    x_error_rel=None,
    y_error_rel=None,
    x_error_cor=None,
    y_error_cor=None,
    x_error_cor_rel=None,
    y_error_cor_rel=None,
    errors_rel_to_model=True,
    limits=None,
    fixed=None,
    constraints=None,
    report=False,
    profile=None,
    save=True,
):
    """
    Built-in function for fitting a model function to xy data.

    Interpretation of x_error, y_error, x_error_rel, and y_error_rel:
    If the input error is a simple float it is broadcast across the entire data vector.
    If the input error is a one-dimensional vector it is interpreted as a pointwise error vector.
    If the input error is a two-dimensional matrix it is interpreted as a covariance matrix.

    Interpretation of x_error_cor, y_error_cor, x_error_cor_rel, and y_error_cor_rel:
    If the input error is a simple float it is broadcast across the entire data vector.
    If the input error is a one-dimensional vector then each individual value is added as
    a separate error that is being broadcast across the entire data vector.

    :param model_function: The model function as a native Python function where the first
        argument denotes the independent *x* variable. Alternatively an already defined
        :py:class:`~kafe2.fit._base.model.ModelFunctionBase` object. Defaults to a straight line.
    :type model_function: typing.Callable
    :param x_data: the x data values for the fit. Must be one-dimensional.
    :type x_data: typing.Sequence[float]
    :param y_data: the y data values for the fit. Must be one-dimensional.
    :type y_data: typing.Sequence[float]
    :param p0: the initial parameter values for the fit.
    :type p0: typing.Sequence[float]
    :param dp0: the initial parameter step size for the fit.
    :type dp0: typing.Sequence[float]
    :param x_error: uncorrelated absolute *x* error.
    :type x_error: float or typing.Sequence[float]
    :param y_error: uncorrelated absolute *y* error.
    :type y_error: float or typing.Sequence[float]
    :param x_error_rel: uncorrelated relative *x* error.
    :type x_error_rel: float or typing.Sequence[float]
    :param y_error_rel: uncorrelated relative *y* error.
    :type y_error_rel: float or typing.Sequence[float]
    :param x_error_cor: correlated absolute *x* error.
    :type x_error_cor: float or typing.Sequence[float]
    :param y_error_cor: correlated absolute *y* error.
    :type y_error_cor: float or typing.Sequence[float]
    :param x_error_cor_rel: correlated relative *x* error.
    :type x_error_cor_rel: float or typing.Sequence[float]
    :param y_error_cor_rel: correlated relative *y* error.
    :type y_error_cor_rel: float or typing.Sequence[float]
    :param errors_rel_to_model: whether the relative *y* errors should be relative to the model.
        Otherwise they are relative to the data.
    :type errors_rel_to_model: bool
    :param limits: limits to be applied to the model parameter. The expected format for each limit
        is an iterable consisting of the parameter name, the lower bound, and then the upper bound.
        An iterable of limits can be passed to limit multiple parameters.
    :type limits: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param fixed: Model parameter to be fixed. The expected format for each parameter is the
        parameter name followed by an optional value to which the parameter should be set prior to
        fixing. An iterable of (name, value) tuples can be passed to fix multiple parameters.
    :type fixed: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param constraints: constraints to be applied to the model parameter. The expected format for
        each constraint is an iterable consisting of the parameter name, the parameter mean, and
        then the parameter uncertainty. An iterable of constraints can be passed to limit multiple
        parameters.
    :type constraints: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param report: whether a report of the data and fit results should be printed to the console.
    :type report: bool
    :param profile: whether the profile likelihood method should be used for asymmetric parameter
        errors and profile/contour plots.
    :type profile: bool
    :param save: whether the fit results should be saved to disk under `results`.
    :type save: bool
    :return: the fit results.
    :rtype: dict
    """
    from kafe2.fit.xy.fit import XYFit

    if model_function is None:
        _fit = XYFit([x_data, y_data])
    else:
        _fit = XYFit([x_data, y_data], model_function)

    def _add_error_to_fit(axis, error, correlated=False, relative=False):
        if error is None:
            return
        error = np.asarray(error)
        _reference = "model" if errors_rel_to_model and axis == "y" and relative else "data"
        if correlated:
            if error.ndim == 0:
                error = np.reshape(error, (1,))
            for _err in error:
                _fit.add_error(axis, _err, correlation=1.0, relative=relative, reference=_reference)
        else:
            if error.ndim == 2:
                _fit.add_matrix_error(axis, error, "cov", relative=relative, reference=_reference)
            else:
                _fit.add_error(axis, error, relative=relative, reference=_reference)

    _add_error_to_fit("x", x_error)
    _add_error_to_fit("y", y_error)
    _add_error_to_fit("x", x_error_rel, relative=True)
    _add_error_to_fit("y", y_error_rel, relative=True)
    _add_error_to_fit("x", x_error_cor, correlated=True)
    _add_error_to_fit("y", y_error_cor, correlated=True)
    _add_error_to_fit("x", x_error_cor_rel, correlated=True, relative=True)
    _add_error_to_fit("y", y_error_cor_rel, correlated=True, relative=True)

    if profile is None:
        profile = x_error is not None or x_error_rel is not None or y_error_rel is not None

    return _fit_wrapper_generic(_fit, p0, dp0, limits, fixed, constraints, report, profile, save)


def plot(
    fits=-1,
    x_label=None,
    y_label=None,
    data_label=None,
    model_label=None,
    error_band_label=None,
    x_range=None,
    y_range=None,
    x_scale=None,
    y_scale=None,
    x_ticks=None,
    y_ticks=None,
    parameter_names=None,
    model_name=None,
    model_expression=None,
    font_scale=1.0,
    legend=True,
    fit_info=True,
    error_band=True,
    extra=None,
    profile=None,
    plot_profile=None,
    show=True,
    save=True,
):
    """
    Plots kafe2 fits.

    :param fits: which kafe2 fits to use for the plot. A positive integer is interpreted as the fit
        with the given index that has been performed (with wrappers) since the program started. A
        negative integer *-n* is interpreted as the last *n* fits. kafe2 fit objects are used
        directly.
    :type fits: int or :py:class:`~kafe2.fit._base.FitBase`
        or Sequence[:py:class:`~kafe2.fit._base.FitBase`]
    :param x_label: the *x* axis label.
    :type x_label: str
    :param y_label: the *y* axis label.
    :type y_label: str
    :param data_label: the data label(s) in the legend.
    :type data_label: str or typing.Sequence[str]
    :param model_label: the model label(s) in the legend (under data label).
    :type model_label: str or typing.Sequence[str]
    :param error_band_label: the error band label(s) in the legend.
    :type error_band_label: str or typing.Sequence[str]
    :param x_range: *x* range for the plot.
    :type x_range: typing.Sequence[float], len(x_range) == 2
    :param y_range: *y* range for the plot.
    :type y_range: typing.Sequence[float], len(y_range) == 2
    :param x_scale: the scale to use for the *x* axis.
    :type x_scale: "linear" or "log"
    :param y_scale: the scale to use for the *y* axis.
    :type y_scale: "linear" or "log"
    :param x_ticks: the ticks at which to show values on the *x* axis.
    :type x_ticks: typing.Sequence[float]
    :param y_ticks: the ticks at which to show values on the *y* axis.
    :type y_ticks: typing.Sequence[float]
    :param parameter_names: custom parameter LaTeX names to display in the plot. The dictionary keys
        are the regular parameter names and the dictionary values are the names to show in the plot.
    :type parameter_names: dict
    :param model_name: the model LaTeX name(s) in the legend (in the mathematical expression of the
        model function).
    :type model_name: str or typing.Sequence[str]
    :param model_expression: the model LaTeX expression(s) in the legend.
    :type model_expression: str or typing.Sequence[str]
    :param legend: whether the legend should be shown.
    :type legend: bool
    :param fit_info: whether the fit information (fit results, goodness of fit) should be shown.
    :type fit_info: bool
    :param error_band: whether the model error band should be shown.
    :type error_band: bool
    :param extra: additional, supplementary plots to show below the main plot.
    :type error_band: "ratio", "residual", or "pull".
    :param profile: whether the profile likelihood method should be used for asymmetric parameter
        errors and profile/contour plots.
    :type profile: bool
    :param plot_profile: whether the profile plots should be created.
    :type plot_profile: bool
    :param show: whether the plots should be shown.
    :type show: bool
    :param save: whether the plots should be saved to disk under `results`.
    :type save: bool
    :param font_scale: multiply font size by this amount.
    :type font_scale: float

    :return: a *kafe2* plot object containing the relevant matplotlib plots.
    :rtype: :py:class:`~kafe2.fit._base.Plot`
    """
    from kafe2 import ContoursProfiler, CustomFit, Plot

    _start_index = _get_file_index()

    _fit_profiles = None
    _file_indices = None
    if isinstance(fits, int):
        if fits >= 0:
            _fit_profiles = [_fit_history[fits]["profile"]]
            _file_indices = [_fit_history[fits]["file_index"]]
            fits = [_fit_history[fits]["fit"]]
        else:
            fits = _fit_history[fits:]
            _fit_profiles = [_f["profile"] for _f in fits]
            _file_indices = [_f["file_index"] for _f in fits]
            fits = [_f["fit"] for _f in fits]
    else:
        try:
            iter(fits)
        except TypeError:
            fits = [fits]
    if profile is None:
        profile = np.any(_fit_profiles)

    if plot_profile is None and _fit_profiles is not None:
        plot_profile = _fit_profiles
    if profile and plot_profile is not None:
        try:
            iter(plot_profile)
        except TypeError:
            plot_profile = [plot_profile for _ in fits]
        for _i, (_f_i, _pp_i) in enumerate(zip(fits, plot_profile)):
            if not _pp_i:
                continue
            _cpf = ContoursProfiler(_f_i)
            _cpf.plot_profiles_contours_matrix(font_scale=font_scale)
            if save:
                _file_index = _start_index + _i if _file_indices is None else _file_indices[_i]
                _cpf.save(f"results/fit-{_file_index:04d}-profile.png", dpi=240)

    fits = [_f for _f in fits if not isinstance(_f, CustomFit)]  # No sensible way to plot this.
    if parameter_names is not None:
        _unused_parameter_names = deepcopy(parameter_names)
        for _f in fits:
            _plns = {_p: _pn for _p, _pn in parameter_names.items() if _p in _f.parameter_names}
            _f.assign_parameter_latex_names(**_plns)
            for _parameter_name in _f.parameter_names:
                _unused_parameter_names.pop(_parameter_name, None)
            for _x_name in _f._model_function.x_name:
                _unused_parameter_names.pop(_x_name, None)
        if _unused_parameter_names:
            warnings.warn(f"Unused parameter names for plot: {_unused_parameter_names}")
    if model_name is not None:
        if isinstance(model_name, str):
            model_name = [model_name for _ in fits]
        for _f, _mn in zip(fits, model_name):
            _f.assign_model_function_latex_name(_mn)
    if model_expression is not None:
        if isinstance(model_expression, str):
            model_expression = [model_expression for _ in fits]
        for _f, _me in zip(fits, model_expression):
            _f.assign_model_function_latex_expression(_me)
    _plot = Plot(fits)
    if x_label is not None:
        _plot.x_label = x_label
    if y_label is not None:
        _plot.y_label = y_label
    if data_label is not None:
        _plot.customize("data", "label", data_label)
    if model_label is not None:
        _plot.customize("model", "label", model_label)
        # XYFit for some reason uses a different keyword:
        try:
            _plot.customize("model_line", "label", model_label)
        except ValueError:
            pass
        # Apply model label to density label in plot if HistFit:
        try:
            _plot.customize("model_density", "label", f"{model_label} density")
        except ValueError:
            pass
    if error_band_label is not None:
        _plot.customize("model_error_band", "label", error_band_label)
    if model_label is not None and error_band_label is None:
        error_band_label = f"{model_label} $\\pm 1 \\sigma$"
        try:
            _plot.customize("model_error_band", "label", error_band_label)
        except ValueError:
            pass
    if not error_band:
        _plot.customize("model_error_band", "label", None)
        _plot.customize("model_error_band", "hide", True)

    if x_range is not None:
        _plot.x_range = x_range
    if y_range is not None:
        _plot.y_range = y_range
    if x_scale is not None:
        _plot.x_scale = x_scale
    if y_scale is not None:
        _plot.y_scale = y_scale
    if x_ticks is not None:
        _plot.x_ticks = x_ticks
    if y_ticks is not None:
        _plot.y_ticks = y_ticks

    if extra not in [None, "ratio", "residual", "pull"]:
        raise ValueError(f"Unknown extra plot: '{extra}'. Available: None, 'ratio', 'residual', 'pull'.")
    if len(fits) > 0:  # Do not plot if only CustomFit.
        _plot.plot(
            legend=legend,
            fit_info=fit_info,
            asymmetric_parameter_errors=profile,
            font_scale=font_scale,
            ratio=extra == "ratio",
            residual=extra == "residual",
            pull=extra == "pull",
        )

        if save:
            for _i, _ in enumerate(fits):
                _file_index = _start_index + _i if _file_indices is None else _file_indices[_i]
                _plot.save(f"results/fit-{_file_index:04d}-plot.png", dpi=240)

    if show:
        _plot.show()

    return _plot


_plot_func = plot


# def plot_xy_data(x_data, y_data, x_error=None, y_error=None, x_error_rel=None, y_error_rel=None,
#                 x_label=None, y_label=None, data_label=None, x_range=None, y_range=None,
#                 x_scale=None, y_scale=None, x_ticks=None, y_ticks=None, show=True, save=True):
#    from kafe2.fit.xy.container import XYContainer
#    _container = XYContainer(x_data, y_data)
#    if x_error is not None:
#        _container.add_error("x", x_error)
#    if y_error is not None:
#        _container.add_error("y", y_error)
#    if x_error_rel is not None:
#        _container.add_error("x", x_error_rel, relative=True)
#    if y_error_rel is not None:
#        _container.add_error("y", y_error_rel, relative=True)
#    plot(_container, x_label=x_label, y_label=y_label, data_label=data_label, x_range=x_range,
#         y_range=y_range, x_scale=x_scale, y_scale=y_scale, x_ticks=x_ticks, y_ticks=y_ticks,
#         show=show, save=save)


def k2Fit(
    func,
    x,
    y,
    sx=None,
    sy=None,
    srelx=None,
    srely=None,
    xabscor=None,
    yabscor=None,
    xrelcor=None,
    yrelcor=None,
    ref_to_model=True,
    constraints=None,
    p0=None,
    dp0=None,
    limits=None,
    plot=True,
    axis_labels=["x-data", "y-data"],
    data_legend="data",
    model_expression=None,
    model_name=None,
    model_legend="model",
    model_band=r"$\pm 1 \sigma$",
    fit_info=True,
    plot_band=True,
    asym_parerrs=True,
    plot_cor=False,
    showplots=True,
    quiet=True,
):
    """
    Legacy function for backwards compatibility with *PhyPraKit*.
    **New code should not use this function.**
    Fits a model to *xy* data and plots the results.

    Interpretation of sx, sy, srelx, and srely:
    If the input error is a simple float it is broadcast across the entire data vector.
    If the input error is a one-dimensional vector it is interpreted as a pointwise error vector.
    If the input error is a two-dimensional matrix it is interpreted as a covariance matrix.

    Interpretation of xabscor, yabscor, xrelcor, and yrelcor:
    If the input error is a simple float it is broadcast across the entire data vector.
    If the input error is a one-dimensional vector then each individual value is added as
    a separate error that is being broadcast across the entire data vector.

    :param func: The model function as a native Python function where the first
        argument denotes the independent *x* variable. Alternatively an already defined
        :py:class:`~kafe2.fit._base.model.ModelFunctionBase` object. Defaults to a straight line.
    :type func: typing.Callable
    :param x: the *x* data values for the fit. Must be one-dimensional.
    :type x: typing.Sequence[float]
    :param y: the *y* data values for the fit. Must be one-dimensional.
    :type y: typing.Sequence[float]
    :param sx: uncorrelated absolute *x* error.
    :type sx: float or typing.Sequence[float]
    :param sy: uncorrelated absolute *y* error.
    :type sy: float or typing.Sequence[float]
    :param srelx: uncorrelated relative *x* error.
    :type srelx: float or typing.Sequence[float]
    :param srely: uncorrelated relative *y* error.
    :type srely: float or typing.Sequence[float]
    :param xabscor: correlated absolute *x* error.
    :type xabscor: float or typing.Sequence[float]
    :param yabscor: correlated absolute *y* error.
    :type yabscor: float or typing.Sequence[float]
    :param xrelcor: correlated relative *x* error.
    :type xrelcor: float or typing.Sequence[float]
    :param yrelcor: correlated relative *y* error.
    :type yrelcor: float or typing.Sequence[float]
    :param ref_to_model: whether the relative *y* errors should be relative to the model.
        Otherwise they are relative to the data.
    :type ref_to_model: bool
    :param constraints: constraints to be applied to the model parameter. The expected format for
        each constraint is an iterable consisting of the parameter name, the parameter mean, and
        then the parameter uncertainty. An iterable of constraints can be passed to limit multiple
        parameters.
    :type constraints: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param p0: the initial parameter values for the fit.
    :type p0: typing.Sequence[float]
    :param dp0: the initial parameter step size for the fit.
    :type dp0: typing.Sequence[float]
    :param limits: limits to be applied to the model parameter. The expected format for each limit
        is an iterable consisting of the parameter name, the lower bound, and then the upper bound.
        An iterable of limits can be passed to limit multiple parameters.
    :type limits: typing.Sequence or typing.Sequence[typing.Union[list, tuple]]
    :param plot: whether the fit results should be plotted.
    :type plot: bool
    :param axis_labels: the labels for the *x* and *y* axis.
    :type axis_labels: typing.Sequence[str]
    :param data_legend: the data label in the legend.
    :type data_legend: str
    :param model_expression: the model LaTeX expression in the legend.
    :type model_expression: str
    :param model_name: the model LaTeX name in the legend (in the mathematical expression of the
        model function).
    :type model_name: str
    :param model_legend: the model label in the legend (under data label).
    :type model_legend: str
    :param model_band: the error band label in the legend.
    :type model_band: str
    :param fit_info: whether the fit information (fit results, goodness of fit) should be shown.
    :type fit_info: bool
    :param plot_band: whether the model error band should be shown.
    :type plot_band: bool
    :param asym_parerrs: whether the profile likelihood method should be used for asymmetric
        parameter errors.
    :type asym_parerrs: bool
    :param plot_cor: whether the profile plots should be created.
    :type plot_cor: bool
    :param showplots: whether the plots should be shown.
    :type showplots: bool
    :param report: whether the report of the data and fit results should be suppressed.
    :type report: bool
    :return: a tuple containing the parameter values, the parameter errors, the parameter
        correlation matrix, and the minimal :math:`\\chi^2` cost function value.
    :rtype: tuple
    """
    xy_fit(
        func,
        x,
        y,
        p0=p0,
        dp0=dp0,
        x_error=sx,
        y_error=sy,
        x_error_rel=srelx,
        y_error_rel=srely,
        x_error_cor=xabscor,
        y_error_cor=yabscor,
        x_error_cor_rel=xrelcor,
        y_error_cor_rel=yrelcor,
        errors_rel_to_model=ref_to_model,
        limits=limits,
        constraints=constraints,
        report=not quiet,
        profile=True,
    )
    if plot:
        _plot_func(
            x_label=axis_labels[0],
            y_label=axis_labels[1],
            data_label=data_legend,
            model_label=model_legend,
            error_band_label=model_band,
            model_name=model_name,
            model_expression=model_expression,
            legend=True,
            fit_info=fit_info,
            error_band=plot_band,
            profile=True,
            plot_profile=plot_cor,
            show=showplots,
        )
    _fit_object = _fit_history[-1]["fit"]
    _parameter_errors = (
        _fit_object.asymmetric_parameter_errors if asym_parerrs else np.stack([-_fit_object.parameter_errors, _fit_object.parameter_errors], axis=-1)
    )
    return (
        _fit_object.parameter_values,
        _parameter_errors,
        _fit_object.parameter_cor_mat,
        _fit_object.goodness_of_fit,
    )
