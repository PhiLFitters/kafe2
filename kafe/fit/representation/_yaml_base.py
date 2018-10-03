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
    def _make_object(cls, yaml_doc, default_type='xy'):
        #strings may be used as shortcuts for frequently used functionality
        if isinstance(yaml_doc, str):
            yaml_doc = cls._process_string(yaml_doc, default_type)
        _overriden_yaml_doc = cls._check_required_keywords_and_override_subspaces(yaml_doc, default_type)
        _object, _leftover_yaml_doc = cls._convert_yaml_doc_to_object(_overriden_yaml_doc.copy())
        if _leftover_yaml_doc:
            raise YamlReaderException("Received unknown or unsupported keywords for constructing a %s object: %s"
                                      % (cls.BASE_OBJECT_TYPE_NAME, _leftover_yaml_doc.keys()))
        return _object
    
    @classmethod
    def _check_required_keywords_and_override_subspaces(cls, yaml_doc, default_type='xy'):
        if not cls._type_required():
            _kafe_object_class = None
        else:
            _object_type = yaml_doc.get('type', None)
            if not _object_type:
                _object_type = default_type
                yaml_doc['type'] = _object_type
                #TODO implicit object type, give out warning?
                #raise YamlReaderException("No type specified for %s object!" % cls.BASE_OBJECT_TYPE_NAME)
            _kafe_object_class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_object_type, None)
            if _kafe_object_class is None:
                raise YamlReaderException("%s type unknown or not supported: %s" 
                                          % (cls.BASE_OBJECT_TYPE_NAME, _object_type))
        
        _override_dict = cls._get_subspace_override_dict(_kafe_object_class)
        for _keyword in list(_override_dict.keys()):
            _value = yaml_doc.pop(_keyword, None)
            if _value:
                _target_namespaces = _override_dict[_keyword]
                if not isinstance(_target_namespaces, list):
                    _target_namespaces = [_target_namespaces]
                for _target_namespace in _target_namespaces:
                    if _target_namespace not in yaml_doc:
                        yaml_doc[_target_namespace] = dict()
                    yaml_doc[_target_namespace][_keyword] = _value
        
        _missing_keywords = [_keyword for _keyword in cls._get_required_keywords(yaml_doc, _kafe_object_class)
                             if _keyword not in yaml_doc]
        if _missing_keywords:
            raise YamlReaderException("Missing required information for reading in a %s object: %s"
                                      % (cls.BASE_OBJECT_TYPE_NAME, _missing_keywords))
            
        return yaml_doc
    
    @classmethod
    def _type_required(cls):
        return True
    
    @classmethod
    def _process_string(cls, string_representation, default_type):
        return dict(type=default_type)
    
    @classmethod
    def _get_required_keywords(cls, yaml_doc, kafe_object_class):
        return []
    
    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        return None, None #format: return <kafe object>, <leftover yaml doc>
    
    @classmethod
    def _get_subspace_override_dict(cls, kafe_object_class):
        return dict()
    
    def read(self):
        with self._ihandle as _h:
            self._yaml_doc = yaml.load(_h, self.LOADER)
        return self._make_object(self._yaml_doc)

