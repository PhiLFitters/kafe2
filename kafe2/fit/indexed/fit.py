from collections import OrderedDict
from copy import deepcopy

from ...tools import print_dict_as_table
from .._base import FitException, FitBase, DataContainerBase
from .container import IndexedContainer
from .._base.cost import CostFunction_Chi2
from .model import IndexedParametricModel, IndexedModelFunction
from .plot import IndexedPlotAdapter
from ..util import collect


__all__ = ['IndexedFit', 'IndexedFitException']


class IndexedFitException(FitException):
    pass


class IndexedFit(FitBase):
    CONTAINER_TYPE = IndexedContainer
    MODEL_TYPE = IndexedParametricModel
    MODEL_FUNCTION_TYPE = IndexedModelFunction
    PLOT_ADAPTER_TYPE = IndexedPlotAdapter
    EXCEPTION_TYPE = IndexedFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}
    _BASIC_ERROR_NAMES = {'data_error', 'model_error', 'data_cov_mat', 'model_cov_mat'}

    def __init__(self,
                 data,
                 model_function,
                 cost_function=CostFunction_Chi2(
                    errors_to_use='covariance',
                    fallback_on_singular=True),
                 minimizer=None,
                 minimizer_kwargs=None):
        """
        Construct a fit of a model to a series of indexed measurements.

        :param data: the measurement values
        :type data: iterable of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe2.fit.indexed.IndexedModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        :param minimizer: the minimizer to use for fitting.
        :type minimizer: None, "iminuit", "tminuit", or "scipy".
        :param minimizer_kwargs: dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """
        super(IndexedFit, self).__init__(
            data=data, model_function=model_function, cost_function=cost_function,
            minimizer=minimizer, minimizer_kwargs=minimizer_kwargs)

    # -- private methods

    def _init_nexus(self):
        super(IndexedFit, self)._init_nexus()

        self._nexus.add_function(
            collect,
            func_name="nuisance_vector"
        )

        # -- initialize nuisance parameters

        # TODO: implement nuisance parameters for indexed data

        self._nexus.add_dependency(
            'model',
            depends_on=(
                'poi_values'
            )
        )

        self._nexus.add_dependency(
            'total_error',
            depends_on=(
                'data_error',
                'model_error',
            )
        )

    def _report_data(self, output_stream, indent, indentation_level):
        output_stream.write(indent * indentation_level + '########\n')
        output_stream.write(indent * indentation_level + '# Data #\n')
        output_stream.write(indent * indentation_level + '########\n\n')

        _data_table_dict = OrderedDict()
        _data_table_dict['Index'] = range(self.data_size)
        _data_table_dict['Data'] = self.data
        if self.has_data_errors:
            _data_table_dict['Data Error'] = self.data_error
            _data_table_dict['Data Total Correlation Matrix'] = self.data_cor_mat

        print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

    def _report_model(self, output_stream, indent, indentation_level):
        # call base method to show header and model function
        super(IndexedFit, self)._report_model(output_stream, indent, indentation_level)
        # print model values at POIs
        _data_table_dict = OrderedDict()
        _data_table_dict['Index'] = range(self.data_size)
        _data_table_dict['Model'] = self.model
        if self.has_model_errors:
            _data_table_dict['Model Error'] = self.model_error
            _data_table_dict['Model Total Correlation Matrix'] = self.model_cor_mat

        print_dict_as_table(_data_table_dict, output_stream=output_stream, indent_level=indentation_level + 1)
        output_stream.write('\n')

    # -- public properties

        # TODO: add 'uncor_cov_mat'
        #for _side in ('data', 'model', 'total'):
        #    self._nexus.add_dependency(
        #        '{}_uncor_cov_mat'.format(_side),
        #        depends_on='{}_cov_mat'.format(_side)
        #    )

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise IndexedFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = IndexedContainer(new_data, dtype=float)
        self._data_container._on_error_change_callback = self._on_error_change

        self._nexus.get('data').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = IndexedParametricModel(
            self._model_function,
            self.parameter_values,
            shape_like=self.data
        )

    # -- public properties

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data
