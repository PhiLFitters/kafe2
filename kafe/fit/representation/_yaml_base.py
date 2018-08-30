import yaml

from kafe.fit.representation._base import DReprWriterMixin, DReprReaderMixin

class YamlWriterException(Exception):
    pass

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

class YamlReaderException(Exception):
    pass

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
    
    @classmethod
    def _make_object(cls, yaml_doc):
        cls._check_required_keywords(yaml_doc)
        _object, _leftover_yaml_doc = cls._convert_yaml_doc_to_object(yaml_doc.copy())
        if _leftover_yaml_doc:
            raise YamlReaderException("Received unknown or unsupported keywords for constructing a %s object: %s"
                                      % (cls.BASE_OBJECT_TYPE_NAME, _leftover_yaml_doc.keys()))
        return _object
    
    @classmethod
    def _check_required_keywords(cls, yaml_doc):
        if cls._type_required():
            _fit_type = yaml_doc.get('type', None)
            if not _fit_type:
                raise YamlReaderException("No type specified for %s object!" % cls.BASE_OBJECT_TYPE_NAME)
            _kafe_object_class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_fit_type, None)
            if _kafe_object_class is None:
                raise YamlReaderException("%s type unknown or not supported: %s" 
                                          % (cls.BASE_OBJECT_TYPE_NAME, _fit_type))
        else:
            _kafe_object_class = None
        _missing_keywords = [_keyword for _keyword in cls._get_required_keywords(yaml_doc, _kafe_object_class)
                             if _keyword not in yaml_doc]
        if _missing_keywords:
            raise YamlReaderException("Missing required keywords for reading in a %s object: %s"
                                      % (cls.BASE_OBJECT_TYPE_NAME, _missing_keywords))
    
    @classmethod
    def _type_required(cls):
        return True
    
    @classmethod
    def _get_required_keywords(cls, yaml_doc, kafe_object_class):
        return []
    
    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        return None, None #format: return <kafe object>, <leftover yaml doc>
    
    def read(self):
        with self._ihandle as _h:
            self._yaml_doc = yaml.load(_h, self.LOADER)
        return self._make_object(self._yaml_doc)

