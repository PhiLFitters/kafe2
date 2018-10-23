import numpy as np

from ...config import kc
from .._base import PlotContainerBase, PlotFigureBase, MultiPlotBase, kc_plot_style
from .._aux import step_fill_between
from . import XYMultiFit



__all__ = ["XYMultiPlotSingular", "XYMultiPlot","XYMultiPlotContainer"]


class XYMultiPlotContainer(PlotContainerBase):
    FIT_TYPE = XYMultiFit

    def __init__(self, xy_multi_fit_object, model_index, n_plot_points_model=100):
        """
        Construct an :py:obj:`XYMultiPlotContainer` for a :py:obj:`~kafe2.fit.multi.XYMultiFit` object:

        :param xy_multi_fit_object: an :py:obj:`~kafe2.fit.multi.XYMultiFit` object
        :type xy_multi_fit_object: :py:obj:`XYMultiFit`
        :param model_index: the index of the underlying model with which this container is associated
        :type model_index: int
        :param n_plot_points_model: the number of points used for the plot
        :tape n_plot_points_model: 100
        """
        super(XYMultiPlotContainer, self).__init__(
            fit_object=xy_multi_fit_object, 
            model_index=model_index
        )
        self._n_plot_points_model = n_plot_points_model

        self._plot_range_x = None

    # -- private methods

    def _compute_plot_range_x(self, pad_coeff=1.1, additional_pad=None):
        if additional_pad is None:
            additional_pad = (0, 0)
        _xmin, _xmax = self._fitter.get_x_range(self._model_index)
        _w = _xmax - _xmin
        self._plot_range_x = (
            0.5 * (_xmin + _xmax - _w * pad_coeff) - additional_pad[0],
            0.5 * (_xmin + _xmax + _w * pad_coeff) + additional_pad[1]
        )

    # -- public properties

    @property
    def data_x(self):
        """data x values"""
        return self._fitter.get_splice(self._fitter.x_data, self._model_index)

    @property
    def data_y(self):
        """data y values"""
        return self._fitter.get_splice(self._fitter.y_data, self._model_index)

    @property
    def data_xerr(self):
        """x error bars for data: ``None`` for :py:obj:`IndexedPlotContainer`"""
        return self._fitter.get_splice(self._fitter.x_error, self._model_index)

    @property
    def data_yerr(self):
        """y error bars for data: total data uncertainty"""
        return self._fitter.get_splice(self._fitter.y_data_error, self._model_index)

    @property
    def model_x(self):
        """x support values for model function"""
        _xmin, _xmax = self.x_range
        return np.linspace(_xmin, _xmax, self._n_plot_points_model)

    @property
    def model_y(self):
        """y values at support points for model function"""
        return self._fitter.eval_model_function(x=self.model_x, model_index=self._model_index)

    @property
    def model_xerr(self):
        """x error bars for model (not used)"""
        return None if np.allclose(self._fitter.x_error, 0) else self._fitter.x_error

    @property
    def model_yerr(self):
        """y error bars for model (not used)"""
        return None if np.allclose(self._fitter.y_data_error, 0) else self._fitter.y_data_error

    @property
    def x_range(self):
        """x plot range"""
        if self._plot_range_x is None:
            self._compute_plot_range_x()
        return self._plot_range_x

    @property
    def y_range(self):
        """y plot range: ``None`` for :py:obj:`XYMultiPlotContainer`"""
        return None

    @property
    def model_function_argument_formatters(self):
        """return model function argument formatters"""
        return self._fitter._model_function.get_argument_formatters(self._model_index)

    # public methods

    def get_formatted_model_function(self, **kwargs):
        """return the formatted model function string"""
        return self._fitter._model_function.formatter.get_formatted(model_index=self._model_index, **kwargs)

    def plot_data(self, target_axis, **kwargs):
        """
        Plot the measurement data to a specified ``matplotlib`` ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` methods ``errorbar`` or ``plot``
        :return: plot handle(s)
        """
        # TODO: how to handle 'data' errors and 'model' errors?
        if self._fitter.has_errors:
            return target_axis.errorbar(self.data_x,
                                        self.data_y,
                                        xerr=self.data_xerr,
                                        yerr=self.data_yerr,
                                        **kwargs)
        else:
            return target_axis.plot(self.data_x,
                                    self.data_y,
                                    **kwargs)

    def plot_model(self, target_axis, **kwargs):
        """
        Plot the model function to a specified matplotlib ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` ``plot`` method
        :return: plot handle(s)
        """
        # TODO: how to handle 'data' errors and 'model' errors?
        return target_axis.plot(self.model_x,
                                self.model_y,
                                **kwargs)

    def plot_model_error_band(self, target_axis, **kwargs):
        """
        Plot an error band around the model model function.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` ``fill_between`` method
        :return: plot handle(s)
        """
        _band_y = self._fitter.get_y_error_band(self._model_index)
        _y = self.model_y
        if self._fitter.has_errors:
            return target_axis.fill_between(
                self.model_x,
                _y - _band_y, _y + _band_y,
                **kwargs)
        else:
            return None  # don't plot error band if fitter input data has no errors...


class XYMultiPlotSingular(PlotFigureBase):

    PLOT_CONTAINER_TYPE = XYMultiPlotContainer
    PLOT_STYLE_CONFIG_DATA_TYPE = 'xy'

    PLOT_SUBPLOT_TYPES = PlotFigureBase.PLOT_SUBPLOT_TYPES.copy()  # don't change original class variable
    PLOT_SUBPLOT_TYPES['model_error_band'] = dict(
        plot_container_method='plot_model_error_band',
    )
    IS_MULTI_PLOT = True

    def __init__(self, fit_objects, model_indices):
        """
        Creates a new `XYMultiPlotSingular` figure containing one or more models.
        
        :param fit_objects: the kafe2 fit objects to be shown in the figure
        :type fit_objects: `XYMultiFit` or iterable thereof
        :param model_indices: the indices of the underlying model functions that the fit_objects
                              are associated with
        :type model_indices: iterable of int
        """
        super(XYMultiPlotSingular, self).__init__(fit_objects=fit_objects, model_indices=model_indices)
        self._plot_range_x = None

class XYMultiPlot(MultiPlotBase):
    
    SINGULAR_PLOT_TYPE = XYMultiPlotSingular
    
    def __init__(self, fit_objects, separate_plots=True):
        """
        Creates a new `XYMultiPlot` object for one or more `XYMultiFit` objects
        
        :param fit_objects: the fit objects for which plots should be created
        :type fit_objects: `XYMultiFit` or an iterable thereof
        :param separate_plots: if ``True``, will create separate plots for each model
                               within each fit object, if ``False`` will create one plot
                               for each fit object
        :type separate_plots: bool
        """
        super(XYMultiPlot, self).__init__(fit_objects, separate_plots=separate_plots)