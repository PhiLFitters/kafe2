import numpy as np
import re
import yaml

from ....core.error import SimpleGaussianError, MatrixGaussianError
from ....fit import HistContainer, IndexedContainer, XYContainer, XYMultiContainer
from .. import _AVAILABLE_REPRESENTATIONS
from ._base import DataContainerDReprBase
from .._base import DReprError, DReprWriterMixin, DReprReaderMixin

__all__ = ["DataContainerYamlReader", "DataContainerYamlWriter"]


class _DataContainerYamlLoader(yaml.Loader):

    # custom directives for reading in matrices

    def matrix(self, node):
        """construct a matrix from a tab/endline separated string"""
        _rows = node.value.split('\n')
        _mat = [_row.split() for _row in _rows if _row]  # last row may be empty -> leave out
        return np.matrix(_mat, dtype=float)

    def symmetric_matrix(self, node):
        """construct a lower triangular matrix from a tab/endline separated string"""
        _rows = node.value.split('\n')[:-1]  # last endline ends the matrix -> no new row
        _mat = [_row.split() for _row in _rows]
        _np_mat = np.matrix(np.zeros((len(_rows), len(_rows))), dtype=float)

        for _i, _row in enumerate(_mat):
            # check shape -> row index must match row length
            if len(_row) != _i + 1:
                raise DReprError("Cannot parse lower triangular matrix: "
                                 "row #{} should have length {}, got {} "
                                 "instead!".format(_i+1, _i+1, len(_row)))
            # fill matrix
            _np_mat[_i, 0:len(_row)] = _row   # fill below diagonal

        _np_mat = _np_mat + np.tril(_np_mat, -1).T  # symmetrize

        return _np_mat

_DataContainerYamlLoader.add_constructor('!matrix', _DataContainerYamlLoader.matrix)
_DataContainerYamlLoader.add_constructor('!symmetric_matrix', _DataContainerYamlLoader.symmetric_matrix)

class _DataContainerYamlDumper(yaml.Dumper):

    # custom directives for writing out matrices

    _regex_space_after_newline = re.compile("\n\s+") # match all spaces immediately after newline
    _regex_format_n_numbers = r"((\S+\s*){{}})"  # match exactly 'n' sequences of non-whitespace followed by optional whitespace

    def matrix(self, numpy_matrix):
        """Represent a matrix as a space/endline separated string.
        Note: symmetric matrices are represented as a lower triangular matrix by defaul
        """

        _is_symmetric = np.allclose(numpy_matrix, numpy_matrix.T)

        # remove brackets
        _string_repr = str(numpy_matrix).replace('[[', ' [').replace(']]', '] ').replace('[', '').replace(']', '').strip()
        # remove all spaces immediately after newline
        _string_repr = re.sub(self.__class__._regex_space_after_newline, "\n", _string_repr)

        # if symmetric, remove everything above upper diagonal
        if _is_symmetric and False:
            _rows = _string_repr.split('\n')
            for _irow, _row in enumerate(_rows):
                _rows[_irow] = re.sub(r"^((\S+\s*){{{}}}).*$".format(_irow+1), r"\1", _row).strip()
            _string_repr = "\n".join(_rows)
            # write lower triangular matrix using the '|'-style
            return self.represent_scalar('!symmetric_matrix', _string_repr, style='|')

        # write full matrix using the '|'-style
        return self.represent_scalar('!matrix', _string_repr, style='|')



# representers for covariance matrices errors
_DataContainerYamlDumper.add_representer(np.matrix, _DataContainerYamlDumper.matrix)


class DataContainerYamlWriter(DReprWriterMixin, DataContainerDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'writer'

    _yaml_error_section_for_axis = {0: 'x_errors',
                                    1: 'y_errors',
                                    None: 'errors'}

    def __init__(self, data_container, output_io_handle):
        super(DataContainerYamlWriter, self).__init__(output_io_handle=output_io_handle,
                                                      data_container=data_container)

    @staticmethod
    def _write_errors_to_yaml(container, yaml_struct):
        # TODO: create public error retrieval interface
        for _err_name, _err_dict in container._error_dicts.items():

            _err_obj = _err_dict['err']
            _err_axis = _err_dict.get('axis', None)

            # get the relevant error section for the current axis, creating it if necessary
            _yaml_section = yaml_struct.setdefault(DataContainerYamlWriter._yaml_error_section_for_axis[_err_axis], [])

            # -- check for relative errors
            _is_relative = _err_obj.relative
            if _is_relative:
                _err_val = _err_obj.error_rel
            else:
                _err_val = _err_obj.error

            # -- collapse identical errors to one float
            if np.allclose(_err_val[0], _err_val):
                _err_val = float(_err_val[0])
            else:
                _err_val = _err_val.tolist()

            # -- handle different error types
            #TODO shouldn't each error be wrapped inside an 'error' namespace?
            if _err_obj.__class__ is SimpleGaussianError:
                _yaml_section.append(dict(
                    name=_err_name,
                    type='simple',
                    error_value=_err_val,
                    relative=_is_relative,
                    correlation_coefficient=_err_obj._corr_coeff  # TODO: public interface for _corr_coeff!
                ))
            elif _err_obj.__class__ is MatrixGaussianError:
                _mtype = _err_obj._matrix_type_at_construction  # TODO: public interface!
                _yaml_section.append(dict(
                    name=_err_name,
                    type='matrix',
                    matrix_type=_mtype,
                    relative=_is_relative,
                ))
                if _mtype == 'covariance':
                    if _is_relative:
                        _yaml_section[-1]['matrix'] = _err_obj.cov_mat_rel #.tolist()
                    else:
                        _yaml_section[-1]['matrix'] = _err_obj.cov_mat #.tolist()
                elif _mtype == 'correlation':
                    _yaml_section[-1]['matrix'] = _err_obj.cor_mat #.tolist()
                    _yaml_section[-1]['error_value'] = _err_val
                else:
                    raise DReprError("Unknown error matrix type '{}'. Valid: 'correlation' or 'covariance'.")
            else:
                raise DReprError("No representation for error type {} "
                                 "implemented!".format(type(_err_obj)))
    
    @staticmethod
    def _make_representation(container):
        _yaml = dict()
        _class = container.__class__

        # -- determine container type
        _type = DataContainerYamlWriter._CONTAINER_CLASS_TO_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Container unknown or not supported: %s" % _class)
        _yaml['type'] = _type

        # -- write representation for container types
        if _class is HistContainer:
            _yaml['bin_edges'] = container.bin_edges.tolist()
            _yaml['raw_data'] = list(map(float, container.raw_data))  # float64 -> float
        elif _class is IndexedContainer:
            _yaml['data'] = container.data.tolist()
        elif _class is XYContainer:
            _yaml['x_data'] = container.x.tolist()
            _yaml['y_data'] = container.y.tolist()
        elif _class is XYMultiContainer:
            for _i in range(container.num_datasets):
                _xy_data_i = container.get_splice(container.data, _i)
                _yaml['x_data_%s' % _i] = _xy_data_i[0].tolist()
                _yaml['y_data_%s' % _i] = _xy_data_i[1].tolist()
        else:
            raise DReprError("Container type unknown or not supported: {}".format(_type))

        # -- write error representation for all container types
        if container.has_errors:
            DataContainerYamlWriter._write_errors_to_yaml(container, _yaml)

        return dict(dataset=_yaml) # wrap inner yaml inside a 'dataset' namespace

    def write(self):
        self._yaml = self._make_representation(self._container)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml, _h, _DataContainerYamlDumper, default_flow_style=False)


class DataContainerYamlReader(DReprReaderMixin, DataContainerDReprBase):
    DREPR_FLAVOR_NAME = 'yaml'
    DREPR_ROLE_NAME = 'reader'

    def __init__(self, input_io_handle):
        super(DataContainerYamlReader, self).__init__(input_io_handle=input_io_handle,
                                                      data_container=None)

    @staticmethod
    def _add_error_to_container(err_type, container_obj, **kwargs):
        # TODO: check kwargs explicitly
        if err_type == 'simple':
            container_obj.add_simple_error(**kwargs)
        elif err_type == 'matrix':
            container_obj.add_matrix_error(**kwargs)
        else:
            raise DReprError("Unknown error type '{}'. "
                             "Valid: {}".format(err_type, ('simple', 'matrix')))

    @staticmethod
    def _make_object(yaml):
        _yaml = yaml['dataset']

        # -- determine container class from type
        _container_type = _yaml['type']
        _class = DataContainerYamlReader._CONTAINER_TYPE_NAME_TO_CLASS.get(_container_type, None)
        if _class is None:
            raise DReprError("Container type unknown or not supported: {}".format(_container_type))

        # -- read in representation for container types
        if _class is HistContainer:
            _bin_edges = _yaml['bin_edges']
            _raw_data = _yaml['raw_data']
            _container_obj = HistContainer(n_bins=len(_bin_edges) - 1,
                                           bin_range=(_bin_edges[0], _bin_edges[-1]),
                                           bin_edges=_bin_edges,
                                           fill_data=_raw_data)
        elif _class is IndexedContainer:
            _data = _yaml['data']
            _container_obj = IndexedContainer(_data)
        elif _class is XYContainer:
            _x_data = _yaml['x_data']
            _y_data = _yaml['y_data']
            _container_obj = XYContainer(_x_data, _y_data)
        elif _class is XYContainer:
            _x_data = _yaml['x_data']
            _y_data = _yaml['y_data']
            _container_obj = XYContainer(_x_data, _y_data)
        elif _class is XYMultiContainer:
            _xy_data = []
            _i = 0 #xy dataset index
            _x_data_i = _yaml.get('x_data_%s' % _i, None)
            _y_data_i = _yaml.get('y_data_%s' % _i, None)
            while _x_data_i is not None and _y_data_i is not None:
                _xy_data.append([_x_data_i, _y_data_i])
                _i += 1
                _x_data_i = _yaml.get('x_data_%s' % _i, None)
                _y_data_i = _yaml.get('y_data_%s' % _i, None)
            _container_obj = XYMultiContainer(_xy_data)
        else:
            raise DReprError("Container type unknown or not supported: {}".format(_container_type))

        # -- process error sources
        if _class in  (XYContainer, XYMultiContainer):
            _xerrs = _yaml.get('x_errors', [])
            _yerrs = _yaml.get('y_errors', [])
            _errs = _xerrs + _yerrs
            _axes = [0] * len(_xerrs) + [1] * len(_yerrs)  # 0 for 'x', 1 for 'y'
        else:
            _errs = _yaml.get('errors', [])
            _axes = [None] * len(_errs)

        # add error sources, if any
        for _err, _axis in zip(_errs, _axes):
            _add_kwargs = dict()
            # translate and check that all required keys are present
            try:
                _err_type = _err['type']

                _add_kwargs['name'] = _err['name']

                if _err_type == 'simple':
                    _add_kwargs['err_val']= _err['error_value']
                    _add_kwargs['correlation']= _err['correlation_coefficient']
                elif _err_type == 'matrix':
                    _add_kwargs['err_matrix'] = _err['matrix']
                    _add_kwargs['matrix_type'] = _err['matrix_type']
                    _add_kwargs['err_val'] = _err.get('error_value', None)  # only mandatory for cor mats; check done later
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

            # add error to data container
            DataContainerYamlReader._add_error_to_container(_err_type, _container_obj, **_add_kwargs)

        return _container_obj

    def read(self):
        with self._ihandle as _h:
            self._yaml = yaml.load(_h, _DataContainerYamlLoader)
        return self._make_object(self._yaml)


# register the above classes in the module-level dictionary
DataContainerYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
DataContainerYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
