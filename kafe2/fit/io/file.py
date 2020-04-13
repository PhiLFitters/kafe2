import abc

from ..io import InputFileHandle, OutputFileHandle


class FileIOException(Exception):
    pass


class FileIOMixin(object):
    """
    Mixin class for kafe2 objects that allows them to be read from/written to files
    using the *kafe2.fit.representation* module.
    """

    def __init__(self, *args, **kwargs):
        super(FileIOMixin, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def _get_base_class(self):
        pass

    @abc.abstractmethod
    def _get_object_type_name(self):
        pass

    @staticmethod
    def _get_file_format(filename, file_format=None):
        _basename_ext = filename.split('.')
        if len(_basename_ext) > 1:
            _basename, _ext = _basename_ext[:-1], _basename_ext[-1]
        else:
            _basename, _ext = _basename_ext[0], None

        if file_format is None and _ext is None:
            raise FileIOException("Cannot detect file format from "
                                  "filename '{}' and no format specified!".format(filename))
        return file_format or _ext  # choose 'format' if specified, otherwise use filename extension

    @classmethod
    def from_file(cls, filename, file_format=None):
        """Read kafe2 object from file"""
        from ..representation import get_reader

        _format = cls._get_file_format(filename=filename, file_format=file_format)
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

    def to_file(self, filename, file_format=None):
        """Write kafe2 object to file"""
        from ..representation import get_writer

        _format = self._get_file_format(filename=filename, file_format=file_format)
        _writer_class = get_writer(self._get_object_type_name(), _format)
        _writer_class(self, OutputFileHandle(filename=filename)).write()
