import abc
import getpass
import datetime
import six


class DReprError(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class GenericDReprBase(object):
    BASE_OBJECT_TYPE_NAME = None
    DREPR_FLAVOR_NAME = None
    DREPR_ROLE_NAME = None

    @classmethod
    def _register_class(cls, global_dict):
        _registered_roles = global_dict.setdefault(cls.BASE_OBJECT_TYPE_NAME, dict())
        _registered_formats_for_role = _registered_roles.setdefault(cls.DREPR_ROLE_NAME, dict())
        _registered_formats_for_role[cls.DREPR_FLAVOR_NAME] = cls


class DReprWriterMixin(object):
    
    DREPR_ROLE_NAME = 'writer'

    """
    A "mixin" class for creating a data representation writer.
    Inheriting from this class in addition to a DRepr class for
    a particular object type adds methods for writing to
    an output stream.

    Derived classes should inherit from :py:class:`DReprWriterMixin` and the
    relevant ``DRepr`` class (in that order).
    """
    def __init__(self, output_io_handle, *args, **kwargs):
        """
        Mixin constructor: sets the output handle

        :param output_io_handle: handle for output stream or file
        :type output_io_handle: :py:class:`~kafe2.fit.io.IOStreamHandle`-derived
        """
        self._ohandle = output_io_handle
        self._representation = None
        super(DReprWriterMixin, self).__init__(*args, **kwargs)

    @staticmethod
    def _make_representation(kafe_object):
        """Implement this method for the different data formats"""
        pass

    def _get_preface_comment(self):
        return '# kafe2 %s %s representation written by %s on %s.\n' % (
            self._kafe_object.__class__.__name__,
            self.DREPR_FLAVOR_NAME,
            getpass.getuser(),
            datetime.datetime.now().strftime('%d.%m.%Y, %H:%M')
        )

    def write(self):
        self._representation = self._make_representation(self._kafe_object)
        with self._ohandle as _h:
            _h.write(self._get_preface_comment())
            _h.write(self._representation)


class DReprReaderMixin(object):
    
    DREPR_ROLE_NAME = 'reader'
    
    """
    A "mixin" class for creating a data representation writer.
    Inheriting from this class in addition to a DRepr class for
    a particular object type adds methods for reading from
    an input stream.

    Derived classes should inherit from :py:class:`DReprReaderMixin` and the
    relevant ``DRepr`` class (in that order).
    """
    def __init__(self, input_io_handle, *args, **kwargs):
        """
        Mixin constructor: sets the input handle

        :param input_io_handle: handle for input stream or file
        :type input_io_handle: :py:class:`~kafe2.fit.io.IOStreamHandle`-derived
        """
        self._ihandle = input_io_handle
        self._object = None
        super(DReprReaderMixin, self).__init__(*args, **kwargs)

    @staticmethod
    def _make_object(representation):
        """
        this method is supposed to create and return an object from the representation.
        It has to be implemented by subclasses.
        """
        pass

    def read(self):
        with self._ihandle as _h:
            self._representation = _h.read()
        return self._make_object(self._representation)
