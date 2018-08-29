import abc


class DReprError(Exception):
    pass


class GenericDReprBase(object):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = None
    DREPR_FLAVOR_NAME = None
    DREPR_ROLE_NAME = None

    @classmethod
    def _register_class(cls, global_dict):
        _registered_roles = global_dict.setdefault(cls.OBJECT_TYPE_NAME, dict())
        _registered_formats_for_role = _registered_roles.setdefault(cls.DREPR_ROLE_NAME, dict())
        _registered_formats_for_role[cls.DREPR_FLAVOR_NAME] = cls


class DReprWriterMixin(object):
    
    DREPR_ROLE_NAME = 'writer'

    """
    A "mixin" class for representing a data representation writer.
    Inheriting from this class in addition to a DRepr class for
    a particular object type adds method for writing to
    an output stream.

    Derived classes should inherit from :py:class:`DReprWriterMixin` and the
    relevant ``DRepr`` class (in that order).
    """
    def __init__(self, output_io_handle, *args, **kwargs):
        """
        Mixin constructor: sets the output handle

        :param output_io_handle: handle for output stream or file
        :type output_io_handle: :py:class:`~kafe.fit.io.IOStreamHandle`-derived
        """
        self._ohandle = output_io_handle
        self._representation = None
        super(DReprWriterMixin, self).__init__(*args, **kwargs)

    @staticmethod
    def _make_representation(kafe_object):
        """Implement this method for the different data formats"""
        pass

    def write(self):
        self._representation = self._make_representation(self._kafe_object)
        with self._ohandle as _h:
            _h.write(self._representation)


class DReprReaderMixin(object):
    
    DREPR_ROLE_NAME = 'reader'
    
    """
    A "mixin" class for representing a data representation writer.
    Inheriting from this class in addition to a DRepr class for
    a particular object type adds method for writing to
    an output stream.

    Derived classes should inherit from :py:class:`DReprWriterMixin` and the
    relevant ``DRepr`` class (in that order).
    """
    def __init__(self, input_io_handle, *args, **kwargs):
        """
        Mixin constructor: sets the input handle

        :param input_io_handle: handle for input stream or file
        :type input_io_handle: :py:class:`~kafe.fit.io.IOStreamHandle`-derived
        """
        self._ihandle = input_io_handle
        self._object = None
        super(DReprReaderMixin, self).__init__(*args, **kwargs)

    @staticmethod
    def _make_object(representation):
        """Implement this method for the different data formats"""
        pass

    def read(self):
        with self._ihandle as _h:
            self._representation = _h.read()
        return self._make_object(self._representation)

