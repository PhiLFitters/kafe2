from __future__ import print_function
import os


class IOStreamHandle(object):
    """
    Very thin wrapper around a Python `file` object, providing an interface for reading/writing to a stream,
    while also allowing it to be used within a context.
    """

    def __init__(self, buffer):
        """
        Create a handle for an existing I/O buffer (e.g. a stream)
        :param buffer: an Python `file` object (or derived)
        """
        self._buffer = buffer

    def __repr__(self):
        return "{}(buffer={})".format(self.__class__.__name__, self._buffer)

    def __enter__(self):
        # don't open stream here: assume already opened
        return self._buffer  # expose the buffer in a context

    def __exit__(self, *args):
        # don't close stream here: assume need to remain open
        pass

    # -- private methods

    # -- public properties

    @property
    def closed(self):
        if self._buffer is not None:
            return self._buffer.closed
        return True  # inexistent buffer -> handle always closed

    @property
    def mode(self):
        if self._buffer is not None:
            return self._buffer.mode
        return None

    # -- public methods

    def flush(self):
        """Flush the underlying buffer"""
        if self._buffer is None:
            raise IOError("Cannot flush buffer: buffer is None")
        return self._buffer.flush()

    def write(self, content, truncate=False):
        """Write to the underlying buffer"""
        if self._buffer is None:
            raise IOError("Cannot perform write: buffer is None")
        if truncate:
            # TODO: use log system
            print("WARNING: truncation of stream requested, ignored")
        return self._buffer.write(content)

    def read(self, *args, **kwargs):
        """Read from the underlying buffer"""
        if self._buffer is None:
            raise IOError("Cannot perform read:  buffer is None")
        return self._buffer.read(*args, **kwargs)

    def seek(self, offset, *args, **kwargs):
        """Seek a position in the underlying buffer"""
        if self._buffer is None:
            raise IOError("Cannot perform seek:  buffer is None")
        return self._buffer.seek(offset, *args, **kwargs)

    def truncate(self, size=None):
        """Truncate the underlying buffer"""
        if self._buffer is None:
            raise IOError("Cannot truncate:      buffer is None")
        # 'size' must be an integer or missing
        _truncate_args = [size] if size is not None else []
        return self._buffer.truncate(*_truncate_args)


class IOFileHandle(IOStreamHandle):
    """
    This class extends `IOStreamHandle` with functionality for reading from and writing to the filesystem.
    The file can be read or written to either directly or by opening the handle in a context:
    `with IOFileHandle(...) as fh: fh.write(...)`
    """

    _VALID_MODES = ['r', 'a']

    def __init__(self, filename, mode):
        super(IOFileHandle, self).__init__(buffer=None)
        self.filename = filename
        self.mode = mode

    def __repr__(self):
        return "{}(filename={}, mode={})".format(self.__class__.__name__, repr(self.filename), repr(self.mode))

    def __enter__(self):
        """for using an IOFileHandle object in a context"""
        if self._buffer is not None:
            # handle already opened: fail
            raise IOError("Cannot open file '{}': already open!".format(self._filename))
        self._buffer = open(self._filename, self._mode)
        return self._buffer

    def __exit__(self, *args):
        """for using an IOFileHandle object in a context"""
        self._buffer.__exit__()
        self._buffer = None

    # -- private methods

    # -- public properties

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        if new_mode not in self.__class__._VALID_MODES:
            raise ValueError("Unknown IOFileHandle mode '{}': expecting one of {}".format(new_mode, self.__class__._VALID_MODES))
        self._mode = new_mode

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, new_filename):
        _abspath = os.path.abspath(new_filename)  # expand relative paths
        if os.path.exists(_abspath) and os.path.isdir(_abspath):
            raise ValueError("Cannot set filename: '{}' is a directory!")
        # TODO: handle inexistent parent directories (?)
        self._filename = new_filename

    # -- public methods

    def write(self, content, truncate=False):
        """Write to the underlying buffer"""
        with self as _fh:
            if truncate:
                _fh.truncate(0)
            _fh.write(content)

    def read(self, *args, **kwargs):
        """Read from the underlying buffer"""
        with self as _fh:
            return _fh.read(*args, **kwargs)


class InputFileHandle(IOFileHandle):
    def __init__(self, filename):
        super(InputFileHandle, self).__init__(filename, mode='r')

    def __repr__(self):
        return "{}(filename={})".format(self.__class__.__name__, repr(self.filename))

    def write(self, content, truncate=False):
        raise IOError("Cannot write to 'InputFileHandle'!")

class OutputFileHandle(IOFileHandle):
    def __init__(self, filename):
        super(OutputFileHandle, self).__init__(filename, mode='a')

    def __repr__(self):
        return "{}(filename={})".format(self.__class__.__name__, repr(self.filename))

    def read(self, *args, **kwargs):
        raise IOError("Cannot read from 'OutputFileHandle'!")


class OutputHandleMultiplexer(IOStreamHandle):
    def __init__(self, *io_handles):
        self._io_handles = io_handles
        self._open_handles = None
        for _fh in self._io_handles:
            if not isinstance(_fh, IOStreamHandle):
                raise ValueError("Cannot initialize OutputHandleMultiplexer: "
                                 "handle object {:r} is of type {}, expected "
                                 "IOStreamHandle!".format(_fh, type(_fh)))

    def __enter__(self):
        self._open_handles = [_fh.__enter__() for _fh in self._io_handles]
        return self

    def __exit__(self, *args):
        for _fh in self._io_handles:
            _fh.__exit__()
        self._open_handles = None

    def write(self, content):
        if self._open_handles is not None:
            for _ofh in self._open_handles:
                _ofh.write(content)
        else:
            with self as _ohm:
                # call a second time from within a context:
                # self._open_handles should no longer be None
                _ohm.write(content)

    def read(self, *args, **kwargs):
        raise IOError("Cannot read from OutputHandleMultiplexer!")

    def flush(self):
        """Flush the underlying buffer"""
        if self._open_handles is None:
            raise IOError("Cannot flush buffers: buffers are not open!")
        for _ofh in self._open_handles:
            _ofh.flush()

    def truncate(self, size=None):
        """Truncate the underlying buffers.
        Note: this fails silently if a truncation operation fails or is not available..."""
        if self._open_handles is None:
            raise IOError("Cannot truncate buffers: buffers are not open!")

        # 'size' must be an integer or missing
        _truncate_args = [size] if size is not None else []
        for _ofh in self._open_handles:
            try:
                _ofh.truncate(*_truncate_args)  # attempt to truncate
            except IOError:
                # for streams that do not support truncation: ignore
                pass


if __name__ == "__main__":
    # TODO: move this to a unit test
    import sys
    # test 1: IOStreamHandle
    print('Instantiating `IOStreamHandle`...')
    _iosh = IOStreamHandle(sys.stdout)
    _iosh.write("1")
    with _iosh as _fh:
        _fh.write("2")
        _fh.write("3\n")

    # test 2: IOFileHandle
    print('Instantiating `OutputFile`')
    _ofh = OutputFileHandle("/tmp/test_IOFileHandle.txt")
    print('Writing...')
    _ofh.write("4", truncate=True)
    with _ofh as _fh:
        _fh.write("5")
        _fh.write("6\n")
    print('...done!')

    print('Trying to read from output stream...')
    try:
        _ofh.read()
    except IOError as e:
        print("...OK! IOError raised:", e)

    _ifh = InputFileHandle("/tmp/test_IOFileHandle.txt")
    print("Reading from file: ")
    print(_ifh.read())

    print('Trying to write to input stream...')
    try:
        _ifh.write("bla")
    except IOError as e:
        print("...OK! IOError raised:", e)

    # test 3: OutputHandleMultiplexer
    print('Instantiating `OutputHandleMultiplexer`')
    _ohm = OutputHandleMultiplexer(_iosh, _ofh)
    print('Writing...')
    _ohm.write("a")
    with _ohm as _fh:
        _fh.write("b")
        _fh.write("c\n")
    print('...done!')

    print("Reading from file: ")
    print(_ifh.read())
