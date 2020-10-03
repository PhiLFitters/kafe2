import numpy as np
import re
import yaml

from ....core.error import SimpleGaussianError, MatrixGaussianError
from ....fit import HistContainer, IndexedContainer, XYContainer, UnbinnedContainer
from .. import _AVAILABLE_REPRESENTATIONS
from ._base import DataContainerDReprBase
from .._base import DReprError
from .._yaml_base import YamlWriterMixin, YamlReaderMixin, YamlReaderException, YamlWriterException

__all__ = ["DataContainerYamlReader", "DataContainerYamlWriter"]


class _DataContainerYamlLoader(yaml.Loader):

    # custom directives for reading in matrices

    def matrix(self, node):
        """construct a matrix from a tab/endline separated string"""
        _rows = node.value.split('\n')
        _mat = [_row.split() for _row in _rows if _row]  # last row may be empty -> leave out
        return np.array(_mat, dtype=float)

    def symmetric_matrix(self, node):
        """construct a lower triangular matrix from a tab/endline separated string"""
        _rows = node.value.split('\n')[:-1]  # last endline ends the matrix -> no new row
        _mat = [_row.split() for _row in _rows]
        _np_mat = np.array(np.zeros((len(_rows), len(_rows))), dtype=float)

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
_DataContainerYamlDumper.add_representer(np.array, _DataContainerYamlDumper.matrix)


class DataContainerYamlWriter(YamlWriterMixin, DataContainerDReprBase):
    DUMPER = _DataContainerYamlDumper

    _yaml_error_section_for_axis = {0: 'x_errors',
                                    1: 'y_errors',
                                    None: 'errors'}

    def __init__(self, data_container, output_io_handle):
        super(DataContainerYamlWriter, self).__init__(output_io_handle=output_io_handle,
                                                      data_container=data_container)

    @staticmethod
    def _write_errors_to_yaml(container, yaml_doc):
        # TODO: create public error retrieval interface
        for _err_name, _err_dict in container._error_dicts.items():

            _err_obj = _err_dict['err']
            _err_axis = _err_dict.get('axis', None)

            # get the relevant error section for the current axis, creating it if necessary
            _yaml_section = yaml_doc.setdefault(DataContainerYamlWriter._yaml_error_section_for_axis[_err_axis], [])

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
    
    @classmethod
    def _make_representation(cls, container):
        """Create a dictionary representing a data container.

        :param container: The data container which will be converted.
        :type container: kafe2.fit._base.DataContainerBase
        """
        _yaml_doc = dict()
        _class = container.__class__

        # -- determine container type
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Container type unknown or not supported: %s" % _class)
        _yaml_doc['type'] = _type

        # -- write representation for container types
        if _class is HistContainer:
            _yaml_doc['bin_edges'] = list(map(float, container.bin_edges))
            if container._manual_heights:
                _yaml_doc['bin_heights'] = list(map(float, container.data))  # float64 -> float
                _yaml_doc['underflow'] = float(container.underflow)
                _yaml_doc['overflow'] = float(container.underflow)
            else:
                _yaml_doc['raw_data'] = list(map(float, container.raw_data))  # float64 -> float
        elif _class is IndexedContainer or _class is UnbinnedContainer:
            _yaml_doc['data'] = container.data.tolist()
        elif _class is XYContainer:
            _yaml_doc['x_data'] = container.x.tolist()
            _yaml_doc['y_data'] = container.y.tolist()
        else:
            raise DReprError("Container type unknown or not supported: {}".format(_type))

        # -- write error representation for all container types
        if container.has_errors:
            cls._write_errors_to_yaml(container, _yaml_doc)

        # write labels for all container types
        if container.label is not None:
            _yaml_doc['label'] = container.label
        _x_axis_label, _y_axis_label = container.axis_labels
        if _x_axis_label is not None:
            _yaml_doc['x_label'] = _x_axis_label
        if _y_axis_label is not None:
            _yaml_doc['y_label'] = _y_axis_label

        return _yaml_doc


class DataContainerYamlReader(YamlReaderMixin, DataContainerDReprBase):
    LOADER = _DataContainerYamlLoader

    def __init__(self, input_io_handle):
        super(DataContainerYamlReader, self).__init__(input_io_handle=input_io_handle,
                                                      data_container=None)

    @staticmethod
    def _add_error_to_container(err_type, container_obj, **kwargs):
        # TODO: check kwargs explicitly
        if err_type == 'simple':
            container_obj.add_error(**kwargs)
        elif err_type == 'matrix':
            container_obj.add_matrix_error(**kwargs)
        else:
            raise DReprError("Unknown error type '{}'. "
                             "Valid: {}".format(err_type, ('simple', 'matrix')))

    @classmethod
    def _get_required_keywords(cls, yaml_doc, container_class):
        if container_class is HistContainer:
            return []
        if container_class is IndexedContainer or container_class is UnbinnedContainer:
            return ['data']
        if container_class is XYContainer:
            return ['x_data', 'y_data']
        raise YamlReaderException("Unknown container type")

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine container class from type
        _container_type = yaml_doc.pop('type')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_container_type)

        # -- read in representation for container types
        if _class is HistContainer:
            _bin_edges = yaml_doc.pop('bin_edges', None)
            _n_bins = yaml_doc.pop('n_bins', None)
            _bin_range = yaml_doc.pop('bin_range', None)
            if not _bin_edges and not (_n_bins and _bin_range):
                raise YamlReaderException("When reading in a histogram dataset either "
                                          "bin_edges or n_bins and bin_range has to be specified!")
            if _bin_edges:
                _n_bins = len(_bin_edges) - 1
                _bin_range = (_bin_edges[0], _bin_edges[-1])
            _raw_data = yaml_doc.pop('raw_data', None)
            _bin_heights = yaml_doc.pop('bin_heights', None)
            if _raw_data and _bin_heights:
                raise YamlReaderException("When reading in a histogram dataset only one out of "
                                          "raw_data and bin_heights can be specified!")
            _container_obj = HistContainer(n_bins=_n_bins,
                                           bin_range=_bin_range,
                                           bin_edges=_bin_edges,
                                           fill_data=_raw_data)
            if _bin_heights:
                _underflow = yaml_doc.pop('underflow', 0)
                _overflow = yaml_doc.pop('overflow', 0)
                _container_obj.set_bins(
                    bin_heights=_bin_heights, underflow=_underflow, overflow=_overflow)
        elif _class is IndexedContainer:
            _data = yaml_doc.pop('data')
            _container_obj = IndexedContainer(_data)
        elif _class is UnbinnedContainer:
            _data = yaml_doc.pop('data')
            _container_obj = UnbinnedContainer(_data)
        elif _class is XYContainer:
            _x_data = yaml_doc.pop('x_data')
            _y_data = yaml_doc.pop('y_data')
            _container_obj = XYContainer(_x_data, _y_data)
        else:
            raise DReprError("Container type unknown or not supported: {}".format(_container_type))

        # get labels for all container types
        _container_obj.label = yaml_doc.pop('label', None)
        _container_obj.axis_labels = (yaml_doc.pop('x_label', None), yaml_doc.pop('y_label', None))

        # -- process error sources
        # errors can be specified as a single float, a list of floats, or a kafe2 error object
        # lists of the above are also valid, if the error object is not a list
        if _class is XYContainer:
            _xerrs = yaml_doc.pop('x_errors', [])
            if not isinstance(_xerrs, list) or (len(_xerrs) > 0 and isinstance(_xerrs[0], float)):
                _xerrs = [_xerrs]
            _yerrs = yaml_doc.pop('y_errors', [])
            if not isinstance(_yerrs, list) or (len(_yerrs) > 0 and isinstance(_yerrs[0], float)):
                _yerrs = [_yerrs]
            _errs = _xerrs + _yerrs
            _axes = [0] * len(_xerrs) + [1] * len(_yerrs)  # 0 for 'x', 1 for 'y'
        else:
            _errs = yaml_doc.pop('errors', [])
            if not isinstance(_errs, list) or (len(_errs) > 0 and isinstance(_errs[0], float)):
                _errs = [_errs]
            _axes = [None] * len(_errs)

        # add error sources, if any
        for _err, _axis in zip(_errs, _axes):
            # if error is a float/int or a list thereof add it as a simple error and don't
            # try to interpret it as a kafe2 error object
            if isinstance(_err, (float, int, list)):
                if _axis is not None:
                    cls._add_error_to_container('simple', _container_obj, err_val=_err, axis=_axis)
                else:
                    cls._add_error_to_container('simple', _container_obj, err_val=_err)
                continue
            elif isinstance(_err, str):
                if _err.endswith("%"):
                    try:
                        _rel_err_percent = float(_err[:-1])
                    except ValueError:
                        raise DReprError("Cannot convert string to relative error: %s" % _err)
                    if _axis is not None:
                        cls._add_error_to_container(
                            'simple', _container_obj, err_val=0.01*_rel_err_percent, relative=True,
                            axis=_axis)
                    else:
                        cls._add_error_to_container(
                            'simple', _container_obj, err_val=0.01*_rel_err_percent, relative=True)
                    continue
                else:
                    raise DReprError("Cannot convert string to error: %s" % _err)

            _add_kwargs = dict()
            # translate and check that all required keys are present
            try:
                _err_type = _err['type']

                _add_kwargs['name'] = _err.get('name')

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

                _add_kwargs['relative'] = _err.get('relative', False)

                # if needed, specify the axis (only for 'xy' containers)
                if _axis is not None:
                    _add_kwargs['axis'] = _axis
            except KeyError as e:
                # KeyErrors mean the YAML is incomplete -> raise
                raise DReprError("Missing required key '%s' for error specification" % e.args[0])

            # add error to data container
            cls._add_error_to_container(_err_type, _container_obj, **_add_kwargs)

        return _container_obj, yaml_doc


# register the above classes in the module-level dictionary
DataContainerYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
DataContainerYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
