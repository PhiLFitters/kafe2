import yaml
import re
import numpy as np

from .._base import DReprError
from ....core.error import SimpleGaussianError, MatrixGaussianError
from ...xy import XYContainer

__all__ = ["add_error_to_container", "write_errors_to_yaml", "process_error_sources",
           "MatrixYamlDumper", "MatrixYamlLoader"]


_yaml_error_section_for_axis = {0: 'x_errors',
                                1: 'y_errors',
                                None: 'errors'}


def add_error_to_container(err_type, container_obj, **kwargs):
    # TODO: check kwargs explicitly
    if err_type == 'simple':
        container_obj.add_error(**kwargs)
    elif err_type == 'matrix':
        container_obj.add_matrix_error(**kwargs)
    else:
        raise DReprError("Unknown error type '{}'. "
                         "Valid: {}".format(err_type, ('simple', 'matrix')))
    return container_obj


def write_errors_to_yaml(container, yaml_doc):
    # TODO: create public error retrieval interface
    for _err_name, _err_dict in container._error_dicts.items():

        _err_obj = _err_dict['err']
        _err_axis = _err_dict.get('axis', None)

        # get the relevant error section for the current axis, creating it if necessary
        _yaml_section = yaml_doc.setdefault(_yaml_error_section_for_axis[_err_axis], [])

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
        # TODO shouldn't each error be wrapped inside an 'error' namespace?
        if _err_obj.__class__ is SimpleGaussianError:
            _yaml_section.append(dict(name=_err_name,
                                      type='simple',
                                      error_value=_err_val,
                                      relative=_is_relative,
                                      correlation_coefficient=_err_obj._corr_coeff,
                                      # TODO: public interface for _corr_coeff!
                                      )
                                 )
        elif _err_obj.__class__ is MatrixGaussianError:
            _mtype = _err_obj._matrix_type_at_construction  # TODO: public interface!
            _yaml_section.append(dict(name=_err_name,
                                      type='matrix',
                                      matrix_type=_mtype,
                                      relative=_is_relative,
                                      )
                                 )
            if _mtype == 'covariance':
                if _is_relative:
                    _yaml_section[-1]['matrix'] = _err_obj.cov_mat_rel  # .tolist()
                else:
                    _yaml_section[-1]['matrix'] = _err_obj.cov_mat  # .tolist()
            elif _mtype == 'correlation':
                _yaml_section[-1]['matrix'] = _err_obj.cor_mat  # .tolist()
                _yaml_section[-1]['error_value'] = _err_val
            else:
                raise DReprError("Unknown error matrix type '{}'. "
                                 "Valid: 'correlation' or 'covariance'.")
        else:
            raise DReprError("No representation for error type {} "
                             "implemented!".format(type(_err_obj)))

    return yaml_doc


def process_error_sources(container_obj, yaml_doc):
    # -- process error sources
    # errors can be specified as a single float, a list of floats, or a kafe2 error object
    # lists of the above are also valid, if the error object is not a list
    if isinstance(container_obj, XYContainer):  # also applies for XYParamModel
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
                container_obj = add_error_to_container('simple', container_obj, err_val=_err,
                                                       axis=_axis)
            else:
                container_obj = add_error_to_container('simple', container_obj, err_val=_err)
            continue
        elif isinstance(_err, str):
            if _err.endswith("%"):
                try:
                    _rel_err_percent = float(_err[:-1])
                except ValueError:
                    raise DReprError("Cannot convert string to relative error: %s" % _err)
                if _axis is not None:
                    container_obj = add_error_to_container('simple', container_obj,
                                                           err_val=0.01 * _rel_err_percent,
                                                           relative=True,
                                                           axis=_axis)
                else:
                    container_obj = add_error_to_container('simple', container_obj,
                                                           err_val=0.01 * _rel_err_percent,
                                                           relative=True)
                continue
            else:
                raise DReprError("Cannot convert string to error: %s" % _err)

        _add_kwargs = dict()
        # translate and check that all required keys are present
        try:
            _err_type = _err['type']

            _add_kwargs['name'] = _err.get('name')

            if _err_type == 'simple':
                _add_kwargs['err_val'] = _err['error_value']
                _add_kwargs['correlation'] = _err['correlation_coefficient']
            elif _err_type == 'matrix':
                _add_kwargs['err_matrix'] = _err['matrix']
                _add_kwargs['matrix_type'] = _err['matrix_type']
                # default None only mandatory for cor mats; check done later
                _add_kwargs['err_val'] = _err.get('error_value', None)
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
        container_obj = add_error_to_container(_err_type, container_obj, **_add_kwargs)

    return container_obj, yaml_doc


class MatrixYamlDumper(yaml.Dumper):
    """Custom directives for writing out matrices"""

    _regex_space_after_newline = re.compile(r"\n\s+")  # match all spaces immediately after newline
    # match exactly 'n' sequences of non-whitespace followed by optional whitespace
    _regex_format_n_numbers = r"((\S+\s*){{}})"

    def matrix(self, numpy_matrix):
        """Represent a matrix as a space/endline separated string.
        Note: symmetric matrices are represented as a lower triangular matrix by default
        """
        _is_symmetric = np.allclose(numpy_matrix, numpy_matrix.T)

        # remove brackets
        _string_repr = str(numpy_matrix).replace('[[', ' [').replace(']]', '] ').replace('[', '').\
            replace(']', '').strip()
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
MatrixYamlDumper.add_representer(np.array, MatrixYamlDumper.matrix)


class MatrixYamlLoader(yaml.Loader):
    """custom directives for reading in matrices"""

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


MatrixYamlLoader.add_constructor('!matrix', MatrixYamlLoader.matrix)
MatrixYamlLoader.add_constructor('!symmetric_matrix', MatrixYamlLoader.symmetric_matrix)
