def xy_fit(x_data, y_data, model_function=None,
           x_error=None, y_error=None, x_error_rel=None, y_error_rel=None,
           correlation_x=0.0, correlation_y=0.0, correlation_x_rel=0.0, correlation_y_rel=0.0,
           limits=[], constraints=[], report=True, show_plots=True, x_label=None, y_label=None,
           filename="fit.png", profile=None):
    from kafe2.fit.xy.container import XYContainer
    from kafe2.fit.xy.fit import XYFit
    from kafe2.fit._base.plot import Plot
    from kafe2.fit.tools.contours_profiler import ContoursProfiler

    _container = XYContainer(x_data, y_data)
    if x_error is not None:
        _container.add_error("x", x_error, correlation=correlation_x)
    if y_error is not None:
        _container.add_error("y", y_error, correlation=correlation_y)
    if x_error_rel is not None:
        _container.add_error("x", x_error_rel, correlation=correlation_x_rel, relative=True)

    if model_function is None:
        _fit = XYFit(_container)
    else:
        _fit = XYFit(_container, model_function)
    if y_error_rel is not None:
        _fit.add_error(
            "y", y_error_rel, correlation=correlation_y_rel, relative=True, reference="model")
    for _limit in limits:
        _fit.limit_parameter(*_limit)
    for _constraint in constraints:
        _fit.add_parameter_constraint(*_constraint)

    if profile is None:
        profile = x_error is not None or x_error_rel is not None or y_error_rel is not None

    _fit.do_fit(asymmetric_parameter_errors=profile)
    if report:
        _fit.report()

    _plot = Plot(_fit)
    if x_label is not None:
        _plot.x_label = x_label
    if y_label is not None:
        _plot.y_label = y_label
    _plot.plot()
    _plot.save(filename)
    if profile:
        _cpf = ContoursProfiler(_fit)
        _cpf.plot_profiles_contours_matrix()
        _cpf.save("profile.png")
    if show_plots:
        _plot.show()
