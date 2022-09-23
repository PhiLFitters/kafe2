import os
from glob import glob
import numpy as np

_fit_history = []


def _get_file_index():
    os.makedirs("results", exist_ok=True)
    _file_index = 0
    _globbed_files = glob(f"results/fit-{_file_index:03d}-*")
    while len(_globbed_files) > 0:
        _file_index += 1
        _globbed_files = glob(f"results/fit-{_file_index:03d}-*")
    return _file_index


def xy_fit(x_data, y_data, model_function=None, p0=None, dp0=None,
           x_error=None, y_error=None, x_error_rel=None, y_error_rel=None,
           x_error_cor=None, y_error_cor=None, x_error_cor_rel=None, y_error_cor_rel=None,
           errors_rel_to_model=True, limits=None, constraints=None, report=False, profile=None,
           save=True):
    from kafe2.fit.xy.fit import XYFit

    if model_function is None:
        _fit = XYFit([x_data, y_data])
    else:
        _fit = XYFit([x_data, y_data], model_function)
    if p0 is not None:
        _fit.set_all_parameter_values(p0)
    if dp0 is not None:
        _fit.parameter_errors = dp0

    def _add_error_to_fit(axis, error, correlated=False, relative=False):
        if error is None:
            return
        error = np.asarray(error)
        _reference = "model" if errors_rel_to_model and axis == "y" and relative else "data"
        if correlated:
            if error.ndim == 0:
                error = np.reshape(error, (1,))
            for _err in error:
                _fit.add_error(axis, _err, correlation=1.0, reference=_reference)
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

    if limits is not None:
        if not isinstance(limits[0], (list, tuple)):
            limits = (limits,)
        for _limit in limits:
            _fit.limit_parameter(*_limit)
    if constraints is not None:
        if not isinstance(constraints[0], (list, tuple)):
            constraints = (constraints,)
        for _constraint in constraints:
            _fit.add_parameter_constraint(*_constraint)

    if profile is None:
        profile = x_error is not None or x_error_rel is not None or y_error_rel is not None

    _fit_result = _fit.do_fit(asymmetric_parameter_errors=profile)
    if report:
        _fit.report(asymmetric_parameter_errors=profile)

    if save:
        _file_index = _get_file_index()
        _fit.save_state(f"results/fit-{_file_index:03d}-results.yml")
        with open(f"results/fit-{_file_index:03d}-report.txt", "w", encoding="utf8") as _f:
            _fit.report(_f, asymmetric_parameter_errors=profile)
    else:
        _file_index = None

    _fit_history.append(dict(fit=_fit, profile=profile, file_index=_file_index))

    return _fit_result


def plot(fits=-1, x_label=None, y_label=None, data_label=None, model_label=None,
         error_band_label=None, x_range=None, y_range=None, x_scale=None, y_scale=None,
         x_ticks=None, y_ticks=None, parameter_names=None, model_name=None, model_expression=None,
         legend=True, fit_info=True, error_band=True, profile=None, plot_profile=None, show=True,
         save=True):
    from kafe2 import Plot, ContoursProfiler

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
    if parameter_names is not None:
        for _f in fits:
            _f.assign_parameter_latex_names(**parameter_names)
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
        _plot.customize("model_line", "label", model_label)
    if error_band_label is not None:
        _plot.customize("model_error_band", "label", error_band_label)
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

    if profile is None:
        profile = np.any(_fit_profiles)
    _plot.plot(legend=legend, fit_info=fit_info, asymmetric_parameter_errors=profile)

    _start_index = _get_file_index()
    if save:
        for _i, _ in enumerate(fits):
            _file_index = _start_index + _i if _file_indices is None else _file_indices[_i]
            _plot.save(f"results/fit-{_file_index:03d}-plot.png", dpi=240)

    if plot_profile is None and _fit_profiles is not None:
        plot_profile = _fit_profiles
    if plot_profile is not None:
        try:
            iter(plot_profile)
        except TypeError:
            plot_profile = [plot_profile for _ in fits]
        for _i, (_f_i, _pp_i) in enumerate(zip(fits, plot_profile)):
            if not _pp_i:
                continue
            _cpf = ContoursProfiler(_f_i)
            _cpf.plot_profiles_contours_matrix()
            if save:
                _file_index = _start_index + _i if _file_indices is None else _file_indices[_i]
                _cpf.save(f"results/fit-{_file_index:03d}-profile.png", dpi=240)
    if show:
        _plot.show()


_plot_func = plot


def plot_xy_data(x_data, y_data, x_error=None, y_error=None, x_error_rel=None, y_error_rel=None,
                 x_label=None, y_label=None, data_label=None, x_range=None, y_range=None,
                 x_scale=None, y_scale=None, x_ticks=None, y_ticks=None, show=True, save=True):
    from kafe2.fit.xy.container import XYContainer
    _container = XYContainer(x_data, y_data)
    if x_error is not None:
        _container.add_error("x", x_error)
    if y_error is not None:
        _container.add_error("y", y_error)
    if x_error_rel is not None:
        _container.add_error("x", x_error_rel, relative=True)
    if y_error_rel is not None:
        _container.add_error("y", y_error_rel, relative=True)
    plot(_container, x_label=x_label, y_label=y_label, data_label=data_label, x_range=x_range,
         y_range=y_range, x_scale=x_scale, y_scale=y_scale, x_ticks=x_ticks, y_ticks=y_ticks,
         show=show, save=save)

def k2Fit(func, x, y, sx=None, sy=None, srelx=None, srely=None, xabscor=None, yabscor=None,
          xrelcor=None, yrelcor=None, ref_to_model=True, constraints=None, p0=None, dp0=None,
          limits=None, plot=True, axis_labels=['x-data', 'y-data'], data_legend='data',
          model_expression=None, model_name=None, model_legend='model',
          model_band=r'$\pm 1 \sigma$', fit_info=True, plot_band=True, asym_parerrs=True,
          plot_cor=False, showplots=True, quiet=True):
    xy_fit(
        x, y, func, p0=p0, dp0=dp0, x_error=sx, y_error=sy, x_error_rel=srelx, y_error_rel=srely,
        x_error_cor=xabscor, y_error_cor=yabscor, x_error_cor_rel=xrelcor, y_error_cor_rel=yrelcor,
        errors_rel_to_model=ref_to_model, limits=limits, constraints=constraints, report=not quiet,
        profile=True
    )
    if plot:
        _plot_func(
            x_label=axis_labels[0], y_label=axis_labels[1], data_label=data_legend,
            model_label=model_legend, error_band_label=model_band, model_name=model_name,
            model_expression=model_expression, legend=True, fit_info=fit_info, error_band=plot_band,
            profile=True, plot_profile=plot_cor, show=showplots
        )
    _fit_object = _fit_history[-1]["fit"]
    _parameter_errors = _fit_object.asymmetric_parameter_errors if asym_parerrs else \
        np.stack([-_fit_object.parameter_errors, _fit_object.parameter_errors], axis=-1)
    return (_fit_object.parameter_values, _parameter_errors, _fit_object.parameter_cor_mat,
            _fit_object.goodness_of_fit)
