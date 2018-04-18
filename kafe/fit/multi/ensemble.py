import numpy as np
import scipy.stats
import six

from ...core.error import CovMat
from .._base import FitEnsembleBase, FitEnsembleException
from ..tools.ensemble import EnsembleVariable, EnsembleVariablePlotter
from .cost import MultiCostFunction_Chi2
from .fit import XYFit

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs


__all__ = ["MultiFitEnsemble"]


def _heuristic_optimal_subplot_grid_size(n_subplots, aspect_ratio_priority=0.5):
    def f2(s, k):
        if n_subplots > s * (s + k):
            return 100000
        return ((s * (s + k) - n_subplots) ** 2 * (1.0 - aspect_ratio_priority)
                + (float(k) / float(s)) ** 2 * (aspect_ratio_priority))

    _optimal_f = np.inf
    _optimal_sk = n_subplots, 0
    for s in six.moves.range(1, n_subplots):
        for k in six.moves.range(0, n_subplots):
            _f = f2(s, k)
            if _f < _optimal_f:
                _optimal_f = _f
                _optimal_sk = s, k

    s, k = _optimal_sk
    return s, s+k


class MultiFitEnsembleException(FitEnsembleException):
    pass


class MultiFitEnsemble(FitEnsembleBase):
    """
    Object for generating ensembles of fits to *xy* pseudo-data generated according to the
    specified uncertainty model.

    After constructing an :py:obj:`~kafe.fit.MultiFitEnsemble` object, an error model should be added
    to it. This is done as for :py:obj:`~kafe.fit.XYFit` objects by using the
    :py:meth:`~kafe.fit.MultiFitEnsemble.add_simple_error` or :py:meth:`~kafe.fit.MultiFitEnsemble.add_matrix_error`
    methods.

    Once an uncertainty model is provided, the fit ensemble can be generated by using the
    :py:meth:`~kafe.fit.MultiFitEnsemble.run` method. This method starts by generating a pseudo-dataset in such a way
    that the empirical distribution of the data corresponds to the specified uncertainty model. It then
    fits the model to the pseudo-data and extracts information from the fit, such as the resulting parameter
    values or the value of the cost function at the minimum. This is repeated a large number of times
    in order to evaluate the whole ensemble in a statistically meaningful way.

    The ensemble result can be visualized by using the :py:meth:`~kafe.fit.MultiFitEnsemble.plot_results` method.

    .. TODO Expand section
    """
    FIT_TYPE = XYFit

    AVAILABLE_STATISTICS = {
        'mean': EnsembleVariable.mean,
        'mean_error': EnsembleVariable.mean_error,
        'std': EnsembleVariable.std,
        'skew': EnsembleVariable.skew,
        'kurtosis': EnsembleVariable.kurtosis,
        'cor_mat': EnsembleVariable.cor_mat,
        'cov_mat': EnsembleVariable.cov_mat,
    }
    _DEFAULT_STATISTICS = {'mean', 'std'}

    def __init__(self, n_experiments, x_support, model_function, model_parameters,
                 cost_function=MultiCostFunction_Chi2(axes_to_use='y', errors_to_use='covariance'),
                 requested_results=None):
        """
        Construct an :py:obj:`~kafe.fit.MultiFitEnsemble` object.

        :param n_experiments: number of pseudoexperiments to perform
        :type n_experiments: int
        :param x_support: *x* values to use as support for calculating the "true" model ("true" *x*)
        :type x_support: iterable of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe.fit.indexed.XYModelFunction` or unwrapped native Python function
        :param model_parameters: parameters of the "true" model
        :type model_parameters: iterable of float
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        :param requested_results: list of result variables to collect for each toy fit
        :type requested_results: iterable of str
        """
        self._n_exp = n_experiments
        self._ref_x_data = np.asarray(x_support, dtype=float)
        self._model_function = model_function
        self._model_parameters = np.asarray(model_parameters)
        self._cost_function = cost_function
        self._n_par = len(self._model_parameters)

        # initialize an `XYFit` object for performing the toy fits
        # need some dummy initial data values in order to initialize a Fit object
        self._ref_y_data = self._model_function(self._ref_x_data, *self._model_parameters)

        # initialize Fit object used for fitting the pseudo-data
        self._toy_fit = XYFit(xy_data=[self._ref_x_data, self._ref_y_data],
                              model_function=self._model_function,
                              cost_function=self._cost_function)

        # set the model parameters of the toy fit to the reference values
        self._set_toy_fit_parameters_to_reference()

        # get reference quantities (y data, covariance matrices...) from toy fit
        self._update_reference_quantities_from_toy_fit()

        # store and validate names of requested ensemble variables
        self._requested_results = requested_results
        if self._requested_results is None:
            self._requested_results = self._DEFAULT_RESULTS
        else:
            # validate list of results requested by user
            _unavailable_results = set(self._requested_results) - set(self.AVAILABLE_RESULTS.keys())
            if _unavailable_results:
                raise ValueError("Requested unavailable result variable(s): %r"
                                 % (_unavailable_results,))

        # initialize `EnsembleVariable` objects to store ensembles
        self._initialize_ensemble_variables()

    def _set_toy_fit_parameters_to_reference(self):
        """set the model parameters of the toy fit to the reference values"""
        self._toy_fit._param_model._model_parameters = self._model_parameters
        self._toy_fit._param_model._pm_calculation_stale = True

    def _generate_pseudodata(self):
        """generate new pseudo-data according to fit error model and commit to data container"""

        if not self._toy_fit.has_errors:
            raise FitEnsembleException("Cannot generate fit ensemble: no error model specified!")

        # -- generate 'x' data
        _x_data = self._ref_x_data.copy()

        if self._toy_fit._data_container.has_x_errors:
            # smear x data according to the total 'x' covariance matrix
            # TODO: only gaussian smearing is implemented -> more?
            _x_jitter =  np.random.multivariate_normal(
                np.zeros_like(_x_data),
                self._ref_x_cov_mat)
            _x_data += _x_jitter

        _y_data = self._toy_fit.eval_model_function(x=_x_data,
                                                    model_parameters=self._model_parameters)

        # smear y data according to the total 'y' covariance matrix
        # TODO: only gaussian smearing is implemented -> more?
        _y_jitter = np.random.multivariate_normal(
            np.zeros_like(_y_data),
            self._ref_y_cov_mat)
        _y_data += _y_jitter

        # update toy fit data container
        self._toy_fit._data_container.x = _x_data
        self._toy_fit._data_container.y = _y_data

    def _gather_results_from_toy_fit(self, i_exp):
        for _var_name in self._requested_results:
            self._ensemble_variables[_var_name].set_value(index=i_exp, variable_value=self._get_var(_var_name))

    def _do_toy_fit(self):
        """run fit with current pseudo-data"""
        self._toy_fit._invalidate_total_error_cache()
        self._toy_fit.do_fit()

    def _get_var(self, var_name):
        """get the value of the result variables for the current fit"""
        return self.AVAILABLE_RESULTS[var_name].fget(self)

    def _initialize_ensemble_variables(self):
        self._ensemble_variables = {}
        self._ensemble_variable_plotters = {}
        if 'y_pulls' in self._requested_results:
            self._ensemble_variables['y_pulls'] = EnsembleVariable(
                ensemble_array=np.zeros((self._n_exp, self.n_dat)),
                distribution=scipy.stats.norm,
                distribution_parameters=dict(loc=0, scale=1)
            )
            self._ensemble_variable_plotters['y_pulls'] = EnsembleVariablePlotter(
                ensemble_variable=self._ensemble_variables['y_pulls'],
                value_ranges=(-3, 3),
                variable_labels=['Pull $y_{%d}$' % (_i,) for _i in six.moves.range(1, self.n_dat+1)]
            )

        if 'x_data' in self._requested_results:
            self._ensemble_variables['x_data'] = EnsembleVariable(
                ensemble_array=np.zeros((self._n_exp, self.n_dat)),
                distribution=scipy.stats.norm,
                distribution_parameters=dict(loc=self._ref_x_data, scale=self._toy_fit.x_total_error)
            )
            self._ensemble_variable_plotters['x_data'] = EnsembleVariablePlotter(
                ensemble_variable=self._ensemble_variables['x_data'],
                value_ranges=np.array([self._ref_x_data - 3 * self._toy_fit.x_total_error,
                                       self._ref_x_data + 3 * self._toy_fit.x_total_error]).T,
                variable_labels=['$x_{%d}$' % (_i,) for _i in six.moves.range(1, self.n_dat+1)]
            )

        if 'y_data' in self._requested_results:
            self._ensemble_variables['y_data'] = EnsembleVariable(
                ensemble_array=np.zeros((self._n_exp, self.n_dat)),
                distribution=scipy.stats.norm,
                distribution_parameters=dict(loc=self._ref_y_data, scale=self._ref_projected_xy_err)
            )
            self._ensemble_variable_plotters['y_data'] = EnsembleVariablePlotter(
                ensemble_variable=self._ensemble_variables['y_data'],
                value_ranges=np.array([self._ref_y_data-3*self._ref_projected_xy_err,
                                       self._ref_y_data+3*self._ref_projected_xy_err]).T,
                variable_labels=['$y_{%d}$' % (_i,) for _i in six.moves.range(1, self.n_dat+1)]
            )

        if 'y_model' in self._requested_results:
            self._ensemble_variables['y_model'] = EnsembleVariable(
                ensemble_array=np.zeros((self._n_exp, self.n_dat)),
                distribution=scipy.stats.norm,
                distribution_parameters=dict(loc=self._ref_y_data, scale=self._ref_projected_xy_err)
            )
            self._ensemble_variable_plotters['y_model'] = EnsembleVariablePlotter(
                ensemble_variable=self._ensemble_variables['y_model'],
                value_ranges=np.array([self._ref_y_data-3*self._ref_projected_xy_err,
                                       self._ref_y_data+3*self._ref_projected_xy_err]).T,
                variable_labels=['$f(x_{%d})$' % (_i,) for _i in six.moves.range(1, self.n_dat+1)]
            )

        if 'parameter_pulls' in self._requested_results:
            self._ensemble_variables['parameter_pulls'] = EnsembleVariable(
                ensemble_array=np.zeros((self._n_exp, self._n_par)),
                distribution=scipy.stats.norm,
                distribution_parameters=dict(loc=0, scale=1)
            )
            self._ensemble_variable_plotters['parameter_pulls'] = EnsembleVariablePlotter(
                ensemble_variable=self._ensemble_variables['parameter_pulls'],
                value_ranges=(-3, 3),
                variable_labels=["Pull ${}$".format(_arg_formatter.latex_name)
                                 for _arg_formatter in self._toy_fit._model_function.argument_formatters]
            )

        if 'cost' in self._requested_results:
            self._ensemble_variables['cost'] = EnsembleVariable(
                ensemble_array=np.zeros((self._n_exp,)),
                distribution=scipy.stats.chi2,  # FIXME: assume chi2 for all cost functions -> change
                distribution_parameters=dict(loc=0, df=self.n_df)
            )
            self._ensemble_variable_plotters['cost'] = EnsembleVariablePlotter(
                ensemble_variable=self._ensemble_variables['cost'],
                value_ranges=(0, 3*self.n_df),
                variable_labels="${}$".format(self._toy_fit._cost_function.formatter.latex_name)
            )

    def _make_figure_gs(self, figsize=(8, 8), nrows=1, ncols=1,
                        left=0.1, bottom=0.1,
                        right=0.9, top=0.9):
        """create a new matplotlib figure with a GridSpec controlling the subplot layout"""
        _fig = plt.figure(figsize=figsize)  # defaults from matplotlibrc
        _gs = gs.GridSpec(nrows=nrows,
                          ncols=ncols,
                          left=left,
                          bottom=bottom,
                          right=right,
                          top=top,
                          wspace=None,
                          hspace=None,
                          height_ratios=None)
        return _fig, _gs

    def _update_reference_quantities_from_toy_fit(self):
        self._ref_y_data = self._toy_fit.eval_model_function(x=self._ref_x_data,
                                                             model_parameters=self._model_parameters)
        self._ref_x_cov_mat = self._toy_fit.x_total_cov_mat
        self._ref_y_cov_mat = self._toy_fit.y_total_cov_mat
        self._ref_projected_xy_cov_mat = self._toy_fit.projected_xy_total_cov_mat
        self._ref_x_err = self._toy_fit.x_total_error
        self._ref_y_err = self._toy_fit.y_total_error
        self._ref_projected_xy_err = self._toy_fit.projected_xy_total_error

    # -- private properties

    @property
    def _x_data(self):
        """property for ensemble variable 'x_data'"""
        return self._toy_fit.x

    @property
    def _parameter_pulls(self):
        """property for ensemble variable 'parameter_pulls'"""
        return (self._toy_fit.parameter_values - self._model_parameters)/self._toy_fit.parameter_errors

    @property
    def _y_data(self):
        """property for ensemble variable 'y_data'"""
        return self._toy_fit.y_data

    @property
    def _y_model(self):
        """property for ensemble variable 'y_model'"""
        return self._toy_fit.y_model

    @property
    def _y_pulls(self):
        """property for ensemble variable 'y_pulls'"""
        return (self._toy_fit.y_data - self._toy_fit.y_model) / self._toy_fit.y_total_error

    @property
    def _cost(self):
        """property for ensemble variable 'cost'"""
        return self._toy_fit.cost_function_value


    # -- public properties

    @property
    def n_exp(self):
        """the number of pseudo-experiments to perform"""
        return self._n_exp

    @property
    def n_par(self):
        """the number of parameters"""
        return self._n_par

    @property
    def n_dat(self):
        """the number of degrees of freedom for the fit"""
        return self._toy_fit._data_container.size

    @property
    def n_df(self):
        """the number of degrees of freedom for the fit"""
        # FIXME: not generally true -> update to handle constrained parameters
        # TODO: not applicable for all cost functions -> find a flexible solution
        return self.n_dat - self.n_par

    # -- public methods

    def add_simple_error(self, axis, err_val, name=None, correlation=0, relative=False, reference='data'):
        self._toy_fit.add_simple_error(axis=axis, err_val=err_val,
                                       name=name, correlation=correlation, relative=relative,
                                       reference=reference)
        self._update_reference_quantities_from_toy_fit()  # recompute reference errors

    # "inherit" docstring
    add_simple_error.__doc__ = XYFit.add_simple_error.__doc__

    def add_matrix_error(self, axis, err_matrix, matrix_type, name=None, err_val=None, relative=False, reference='data'):
        self._toy_fit.add_matrix_error(axis=axis, err_matrix=err_matrix,
                                       matrix_type=matrix_type, name=name, err_val=err_val,
                                       relative=relative, reference=reference)
        self._update_reference_quantities_from_toy_fit()  # recompute reference errors

    # "inherit" docstring
    add_matrix_error.__doc__ = XYFit.add_matrix_error.__doc__

    def run(self):
        """Perform the pseudo-experiments. Retrieve and store the requested fit result variables."""
        self._set_toy_fit_parameters_to_reference()
        self._update_reference_quantities_from_toy_fit()
        self._initialize_ensemble_variables()
        for _i_exp in six.moves.range(self.n_exp):
            self._generate_pseudodata()
            self._do_toy_fit()
            self._gather_results_from_toy_fit(_i_exp)

    def get_results(self, *results):
        """
        Return a dictionary containing the ensembles of result variables.

        :param results: names of result variables to retrieve
        :type results: iterable of str. Calling without arguments retrieves *all* collected results.
        :return: dict
        """
        if not results:
            results = self._requested_results
        else:
            # validate list of results requested by user
            _unavailable_results = set(self._requested_results) - set(self.AVAILABLE_RESULTS.keys())
            if _unavailable_results:
                raise ValueError("Requested unavailable result variable(s): %r"
                                 % (_unavailable_results,))

        _dict_to_return = dict()
        for _result_name in results:
            _var = self._ensemble_variables.get(_result_name, None)
            if _var is None:
                raise FitEnsembleException("Cannot retrieve result '{}': "
                                           "variable not collected!".format(_result_name))
            _dict_to_return[_result_name] = _var.values

        return _dict_to_return

    def get_results_statistics(self, results='all', statistics='all'):
        """
        Return a dictionary containing statistics (e.g. mean) of the result variables.

        :param results: names of retrieved fit variable for which to return statistics
        :type results: iterable of str or ``'all'`` (get statistics for all retrieved variables)
        :param statistics: names of statistics to retrieve for each result variable
        :type statistics: iterable of str or ``'all'`` (get all statistics for each retrieved variable)
        :return: dict
        """
        if results == 'all':
            results = self._requested_results

        if statistics == 'all':
            statistics = self.__class__._DEFAULT_STATISTICS

        _dict_to_return = dict()
        for _result_name in results:
            #_result_array = self._result_array_dicts.get(_result_name, None)
            _result_variable = self._ensemble_variables.get(_result_name, None)
            if _result_variable is None:
                raise FitEnsembleException("Cannot retrieve statistics for result "
                                           "variable '{}': variable not collected!".format(_result_name))

            _current_result_dict = _dict_to_return[_result_name] = dict()

            # calculate and store statistics
            for _stat_name in statistics:
                _stat_unbound_method = self.__class__.AVAILABLE_STATISTICS.get(_stat_name, None)
                if _stat_unbound_method is None:
                    raise FitEnsembleException(
                        "Unknown statistic '%s' requested!" % (_stat_name,))
                _stat = _stat_unbound_method.__get__(_result_variable, EnsembleVariable)
                _current_result_dict[_stat_name] = _stat

        return _dict_to_return

    def plot_result_distributions(self, results='all',
                                  show_legend=True):
        """
        Make plots with histograms of the requested fit variable values across all pseudo-experiments.

        :param results: names of retrieved fit variable for which to generate plots
        :type results: iterable of str or ``'all'`` (make plots for all retrieved variables)
        :param show_legend: if ``True``, show a plot legend on each figure
        :type show_legend: bool
        """
        if results == 'all':
            results = self._requested_results
        else:
            # validate list of results requested by user
            _unavailable_results = set(self._requested_results) - set(self.AVAILABLE_RESULTS.keys())
            if _unavailable_results:
                raise ValueError("Requested unavailable result variable(s): %r"
                                 % (_unavailable_results,))

        for _result_name in results:
            _result_variable = self._ensemble_variables.get(_result_name, None)

            if _result_variable is None:
                raise FitEnsembleException("Cannot plot result for variable '%s': "
                                           "variable not collected!" % (_result_name,))

            _result_variable_plotter = self._ensemble_variable_plotters.get(_result_name, None)

            if _result_variable_plotter is None:
                raise FitEnsembleException("Cannot plot result for variable '%s': "
                                           "no plotter defined!" % (_result_name,))

            # -- decide how to lay out plots depending on the result variable dimensionality
            if _result_variable.ndim == 0:
                # if the ensemble variable is a scalar,
                # plot it into a single `Axes` object
                _fig, _gs = self._make_figure_gs(figsize=(8, 8), nrows=1, ncols=1)
                _ax = plt.subplot(_gs[0, 0])
                # call the plotting routine on the axes grid
                _plot_result_dict = _result_variable_plotter.plot_hist(_ax)

            elif _result_variable.ndim == 1:
                # if the ensemble variable is a one-dimensional vector,
                # plot each entry into a separate `Axes` object and display
                # them in a grid-like layout
                _nplots = int(_result_variable.shape[0])
                _nrows, _ncols = _heuristic_optimal_subplot_grid_size(_nplots, aspect_ratio_priority=0.8)
                _fig, _gs = self._make_figure_gs(figsize=(8, 8), nrows=_nrows, ncols=_ncols)

                # create an array 'a' with a[i, j] = [i, j]
                _axes_grid = np.dstack((np.meshgrid(np.arange(_nrows), np.arange(_ncols))))
                # replace [i, j] by the `Axes` object for _gs[i, j] -> array of `Axes`
                _axes_grid = np.apply_along_axis(
                    lambda irow_icol: plt.subplot(_gs[irow_icol[0], irow_icol[1]]) if irow_icol[0]*_ncols+irow_icol[1] < _nplots else None,
                    -1, _axes_grid)
                # reshape the `Axes` array to match the variable shape
                _axes_grid = _axes_grid.T.flatten()[:_result_variable.shape[0]]
                # call the plotting routine on the axes grid
                _plot_result_dict = _result_variable_plotter.plot_hist(_axes_grid)

            elif _result_variable.ndim == 2:
                # if the ensemble variable is a two-dimensional vector,
                # plot the (i,j)-th entry into a an `Axes` object at the
                # (i,j)-th position in a grid

                _nrows = _result_variable.shape[0]
                _ncols = _result_variable.shape[1]

                _fig, _gs = self._make_figure_gs(figsize=(8, 8), nrows=_nrows, ncols=_ncols)

                # create an array 'a' with a[i, j] = [i, j]
                _axes_grid = np.dstack(reversed(np.meshgrid(np.arange(_nrows), np.arange(_ncols))))
                # replace [i, j] by the `Axes` object for _gs[i, j] -> array of `Axes`
                _axes_grid = np.apply_along_axis(
                    lambda irow_icol: plt.subplot(_gs[irow_icol[0], irow_icol[1]]),
                    -1, _axes_grid)
                # do not reshape _axes_grid -> its shape already matches variable shape

                # call the plotting routine on the axes grid
                _plot_result_dict = _result_variable_plotter.plot_hist(_axes_grid)
            else:
                # cannot plot variables with 3 or more dimensions...
                raise FitEnsembleException("Cannot plot result for variable '%s': variable entry dimensionality "
                                           "too high (%d)!" % (_result_name, _result_variable.ndim))

            if show_legend:
                _fig.legend(_plot_result_dict['legend_handles'],
                            _plot_result_dict['legend_labels'], loc='lower center')
                # add extra space at figure bottom for legend
                _figure_extra_bottom = 0.05 * len(_plot_result_dict['legend_labels'])
            else:
                # no extra space at figure bottom
                _figure_extra_bottom = 0.0

            _fig.canvas.set_window_title(_result_name)

            _gs.tight_layout(_fig,
                             pad=0.0, w_pad=0, h_pad=-0.2,
                             rect=(0.01, 0.02+_figure_extra_bottom, 0.98, 0.98))

        return _plot_result_dict

    def plot_result_scatter(self, results='all',
                                  show_legend=True):
        """
        Make plots with histograms of the requested fit variable values across all pseudo-experiments.

        :param results: names of retrieved fit variable for which to generate plots
        :type results: iterable of str or ``'all'`` (make plots for all retrieved variables)
        :param show_legend: if ``True``, show a plot legend on each figure
        :type show_legend: bool
        """
        if results == 'all':
            results = self._requested_results
        else:
            # validate list of results requested by user
            _unavailable_results = set(self._requested_results) - set(self.AVAILABLE_RESULTS.keys())
            if _unavailable_results:
                raise ValueError("Requested unavailable result variable(s): %r"
                                 % (_unavailable_results,))

        for _result_name in results:
            _result_variable = self._ensemble_variables.get(_result_name, None)

            if _result_variable is None:
                raise FitEnsembleException("Cannot plot result for variable '%s': "
                                           "variable not collected!" % (_result_name,))

            _result_variable_plotter = self._ensemble_variable_plotters.get(_result_name, None)

            if _result_variable_plotter is None:
                raise FitEnsembleException("Cannot plot result for variable '%s': "
                                           "no plotter defined!" % (_result_name,))

            # -- decide how to lay out plots depending on the result variable dimensionality
            if _result_variable.ndim != 1:
                raise ValueError()

            # if the ensemble variable is a one-dimensional vector,
            # plot each entry into a separate `Axes` object and display
            # them in a grid-like layout
            _nrows = _ncols = int(_result_variable.shape[0])
            if _nrows <= 1:
                raise FitEnsembleException("Cannot create scatter plot for result variable '%s': "
                                           "vector has less than two entries!" % (_result_name,))
            _fig, _gs = self._make_figure_gs(figsize=(8, 8), nrows=_nrows-1, ncols=_ncols-1)

            # create an array 'a' with a[i, j] = [i, j]
            _axes_grid = np.dstack((np.meshgrid(np.arange(_nrows), np.arange(_ncols))))
            # replace [i, j] by the `Axes` object for _gs[i, j] -> array of `Axes`
            _axes_grid = np.apply_along_axis(
                lambda irow_icol: plt.subplot(_gs[irow_icol[0] - 1, irow_icol[1]]) if irow_icol[0] > irow_icol[1] else None,
                -1, _axes_grid)

            # call the plotting routine on the axes grid
            _plot_result_dict = _result_variable_plotter.plot_scatter(_axes_grid)

            if show_legend:
                _fig.legend(_plot_result_dict['legend_handles'],
                            _plot_result_dict['legend_labels'], loc='lower center')
                # add extra space at figure bottom for legend
                _figure_extra_bottom = 0.05 * len(_plot_result_dict['legend_labels'])
            else:
                # no extra space at figure bottom
                _figure_extra_bottom = 0.0

            _fig.canvas.set_window_title(_result_name)

            _gs.tight_layout(_fig,
                             pad=0.0, w_pad=0, h_pad=-0.2,
                             rect=(0.01, 0.02+_figure_extra_bottom, 0.98, 0.98))

        return _plot_result_dict


    AVAILABLE_RESULTS = {
        'parameter_pulls': _parameter_pulls,
        'x_data': _x_data,
        'y_pulls': _y_pulls,
        'cost': _cost,
        'y_data': _y_data,
        'y_model': _y_model,
    }
    _DEFAULT_RESULTS = {'y_pulls', 'parameter_pulls', 'cost'}