from __future__ import print_function

import contextlib
import numpy as np
import six
import sys

from string import ascii_letters


@contextlib.contextmanager
def numpy_print_options(*args, **kwargs):
    """Context for fine-tuning the printout of numpy arrays"""
    _saved_options = np.get_printoptions()
    # set the specified print options
    np.set_printoptions(*args, **kwargs)
    try:
        # execute the context body
        yield
    finally:
        np.set_printoptions(**_saved_options)


_ALPHANUMERIC = np.array(list(ascii_letters) + list("0123456789"))
def random_alphanumeric(size):
    return "".join(np.random.choice(_ALPHANUMERIC, size=size))


def print_dict_recursive(dct, output_stream=sys.stdout, indent_level=0, indent_width=4, indent_char=' ', filter_func=lambda x: True):
    _max_key_len = max([len(_key) for _key in dct.keys()])
    _key_format = "{{:{0}}}".format(_max_key_len + 5)

    _indent_prefix = indent_char * indent_width * indent_level

    for k, v in six.iteritems(dct):
        if isinstance(v, dict):
            output_stream.write(_indent_prefix + _key_format.format(k+':') + '\n')
            output_stream.flush()
            print_dict_recursive(v, indent_level=indent_level+1,  indent_width=indent_width, indent_char=indent_char, filter_func=filter_func)
        else:
            if filter_func(v):
                _content = "{}".format(v)
                if '\n' in _content:
                    _content = '\n' + _content
                    _content = _content.replace('\n', '\n{}{}'.format(_indent_prefix, indent_char * indent_width))
                output_stream.write(_indent_prefix + _key_format.format(k+':') + _content + '\n')
                output_stream.flush()


def print_dict_as_table(dct, output_stream=sys.stdout, sep='  ', cell_format='%.4g',
                        indent_level=0, indent_width=4, indent_char=' '):

    _indent_prefix = indent_char * indent_width * indent_level

    _column_heads = list(dct.keys())
    _column_arrays = list(dct.values())

    # apply formatting to (numeric) cell contents
    for _icol, _col in enumerate(_column_arrays):
        for _irow, _cell in enumerate(_col):
            try:
                _col[_irow] = cell_format % (_cell,)
            except TypeError:
                pass

    _n_rows = len(_column_arrays[0])
    _col_has_n_rows = [_n_rows == len(_arr) for _arr in _column_arrays]
    if False in _col_has_n_rows:
        _wrong_size_columns = [_ch for _i, _ch in enumerate(_column_heads) if not _col_has_n_rows[_i]]
        raise ValueError("All column sizes must be equal. Offending columns: {}".format(_wrong_size_columns))

    _column_formats = []
    _column_widths = []
    _column_show_heads = []
    for _column_head, _column_array in zip(_column_heads, _column_arrays):
        _column_width = max([len("{}".format(str(_cell).strip())) for _cell in _column_array])
        _show_head = not _column_head.startswith('_')
        _column_show_heads.append(_show_head)
        if _show_head:
            _column_width = max(len(_column_head), _column_width)
        _column_formats.append("{{:<{0}}}".format(_column_width))
        #_column_formats.append("{}")
        _column_widths.append(_column_width)

    # transpose table into row-first order
    _table = []
    for _irow in six.moves.range(_n_rows):
        _table.append([])
    for _icol, _col in enumerate(_column_arrays):
        for _irow, _cell in enumerate(_col):
            #_table[_irow].append(np.asarray(_cell))
            _table[_irow].append(_cell)

    # print table header
    for _ihead, (_column_head, _column_format, _column_show_head) in enumerate(zip(_column_heads, _column_formats, _column_show_heads)):
        _column_heads[_ihead] = _column_format.format(_column_head if _column_show_head else "")

    with numpy_print_options(
        #formatter={'float': cell_format.format},
        precision=6,
        suppress=True,  # suppress printing small floating point values
        linewidth=100000  # prevent line breaks in matrix by setting this to a high value
    ):

        output_stream.write(_indent_prefix + sep.join(_column_heads) + '\n')

        # print head separator
        _head_seps = ["="*_w if _column_show_heads[_icol] else " "*_w for _icol, _w in enumerate(_column_widths)]
        output_stream.write(_indent_prefix + sep.join(_head_seps) + '\n')

        # print table body
        for _irow, _row in enumerate(_table):
            for _icol, (_cell, _column_format) in enumerate(zip(_row, _column_formats)):
                try:
                    _row[_icol] = _column_format.format(str(_cell).strip())
                except TypeError:
                    raise
            output_stream.write(_indent_prefix + sep.join(_row) + '\n')


def get_compact_representation(parameter_names, parameter_values, parameter_errors, parameter_cor_mat,
                               asymmetric_parameter_errors=None, line_prefix='# ', table_format='rst'):
    assert (len(parameter_names) == len(parameter_values) == len(parameter_errors) == parameter_cor_mat.shape[0]
            == parameter_cor_mat.shape[1])
    try:
        import tabulate
        if asymmetric_parameter_errors is None:
            _headers = ['Par name', 'Par val', 'Par err', 'Par cor mat']
        else:
            _headers = ['Par name', 'Par val', 'Par err parabolic', 'Par err down', 'Par err up', 'Par cor mat']
        _data = []

        _reduced_cor_mat = [_cor_mat_row[:_i] for _i, _cor_mat_row in enumerate(parameter_cor_mat.tolist())]
        _cor_mat_row_strs = tabulate.tabulate(tabular_data=_reduced_cor_mat, tablefmt='plain', floatfmt='.2g').split('\n')

        for _i, (_par_name, _par_val, _par_err, _cor_mat_row_str) in enumerate(
                zip(parameter_names, parameter_values, parameter_errors, _cor_mat_row_strs)):
            _row = [_par_name]
            if np.isnan(_par_err) or _par_err == 0.0:  # check if parameter is fixed
                _row.append(_par_val)
                _row.append('fixed')
            else:  # parameter is not fixed, round and add errors
                _sig_fig_err = max(2, -int(np.log10(np.abs(_par_err))) + 1)
                _sig_fig_val = max(_sig_fig_err, -int(np.log10(np.abs(_par_val))) + 2)
                _row.append(round(_par_val, _sig_fig_val))
                _row.append(round(_par_err, _sig_fig_err))
            if asymmetric_parameter_errors is not None:
                for _err in asymmetric_parameter_errors[_i]:  # iterate over up and down error
                    if np.isnan(_err):  # parameter is fixed, no error available
                        _row.append('N/A')
                    else:
                        _sig_err = max(2, -int(np.log10(np.abs(_err))) + 1)
                        _row.append(round(_err, _sig_err))
            _row.append(_cor_mat_row_str)
            _data.append(_row)

        _representation = tabulate.tabulate(tabular_data=_data, headers=_headers, tablefmt=table_format)
        _representation = _representation.replace('\n', '\n' + line_prefix)
        _representation = line_prefix + _representation + '\n'
    except ImportError:
        _representation = '# ERROR: Could not create human-readable table for model parameters because ' \
                          'tabulate is not installed.\n'
    return _representation
