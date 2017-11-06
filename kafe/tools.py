from __future__ import print_function

import contextlib
import numpy as np
import six
import sys


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