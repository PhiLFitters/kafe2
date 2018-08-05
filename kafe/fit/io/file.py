import abc
from ..io import InputFileHandle, OutputFileHandle

class FileIOException(Exception):
    pass

class FileIOMixin(object):
    """
    Mixin class for kafe objects that allows them to be read from/written to files
    using the *kafe.fit.representation* module.
    """
    def __init__(self, *args, **kwargs):
        super(FileIOMixin, self).__init__(*args, **kwargs)
        
    @abc.abstractmethod
    def _get_base_class(self):
        pass
    
    @abc.abstractmethod
    def _get_object_type_name(self):
        pass
        
    @classmethod
    def from_file(cls, filename, format=None):
        """Read kafe object from file"""
        from ..representation import get_reader

        _basename_ext = filename.split('.')
        if len(_basename_ext) > 1:
            _basename, _ext = _basename_ext[:-1], _basename_ext[-1]
        else:
            _basename, _ext = _basename_ext[0], None

        if format is None and _ext is None:
            raise FileIOException("Cannot detect file format from "
                                  "filename '{}' and no format specified!".format(filename))
        else:
            _format = format or _ext  # choose 'format' if specified, otherwise use filename extension

        _reader_class = get_reader(cls._get_object_type_name(), _format)
        _object = _reader_class(InputFileHandle(filename=filename)).read()

        # check if the container is the right type (do not check if calling from DataContainerBase)
        if not _object.__class__ == cls and not cls == cls._get_base_class():
            raise FileIOException("Cannot import '{}' from file '{}': file contains wrong {} type '{}'!".format(
                cls.__name__, 
                filename,
                cls._get_object_type_name(),
                _object.__class__.__name__
            ))
        return _object

    def to_file(self, filename, format=None):
        """Write kafe object to file"""
        from ..representation import get_writer

        _basename_ext = filename.split('.')
        if len(_basename_ext) > 1:
            _basename, _ext = _basename_ext[:-1], _basename_ext[-1]
        else:
            _basename, _ext = _basename_ext[0], None

        if format is None and _ext is None:
            raise FileIOException("Cannot detect file format from "
                                  "filename '{}' and no format specified!".format(filename))
        else:
            _format = format or _ext  # choose 'format' if specified, otherwise use filename extension

        _writer_class = get_writer(self._get_object_type_name(), _format)
        _writer_class(self, OutputFileHandle(filename=filename)).write()