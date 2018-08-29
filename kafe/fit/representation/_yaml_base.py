import yaml

from kafe.fit.representation._base import DReprWriterMixin, DReprReaderMixin

class YamlWriterMixin(DReprWriterMixin):
 
    DREPR_FLAVOR_NAME = 'yaml'
    DUMPER = yaml.Dumper
 
    """
    A "mixin" class for creating a *yaml* representation writer.
    Inheriting from this class in addition to a DRepr class for
    a particular object type adds methods for writing a yaml document 
    to an output stream.

    Derived classes should inherit from :py:class:`YamlWriterMixin` and the
    relevant ``DRepr`` class (in that order).
    """
    def write(self):
        self._yaml_doc = self._make_representation(self._kafe_object)
        with self._ohandle as _h:
            try:
                # try to truncate the file to 0 bytes
                _h.truncate(0)
            except IOError:
                # if truncate not available, ignore
                pass
            yaml.dump(self._yaml_doc, _h, default_flow_style=False)

class YamlReaderMixin(DReprReaderMixin):

    DREPR_FLAVOR_NAME = 'yaml'
    LOADER = yaml.Loader #TODO SafeLoader

    """
    A "mixin" class for creating a *yaml* representation writer.
    Inheriting from this class in addition to a DRepr class for
    a particular object type adds methods for writing to
    an output stream.

    Derived classes should inherit from :py:class:`YamlReaderMixin` and the
    relevant ``DRepr`` class (in that order).
    """
    def read(self):
        with self._ihandle as _h:
            self._yaml_doc = yaml.load(_h, self.LOADER) 
        return self._make_object(self._yaml_doc)

