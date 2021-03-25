from ....fit import HistContainer, IndexedContainer, XYContainer, UnbinnedContainer
from .. import _AVAILABLE_REPRESENTATIONS
from ._base import DataContainerDReprBase
from .._base import DReprError
from .._yaml_base import YamlWriterMixin, YamlReaderMixin, YamlReaderException
from ..error import common_error_tools

__all__ = ["DataContainerYamlReader", "DataContainerYamlWriter"]


class DataContainerYamlWriter(YamlWriterMixin, DataContainerDReprBase):
    DUMPER = common_error_tools.MatrixYamlDumper

    def __init__(self, data_container, output_io_handle):
        super(DataContainerYamlWriter, self).__init__(output_io_handle=output_io_handle,
                                                      data_container=data_container)

    @classmethod
    def _make_representation(cls, container):
        """Create a dictionary representing a data container.

        :param container: The data container which will be converted.
        :type container: kafe2.fit._base.DataContainerBase
        """
        _yaml_doc = dict()
        _class = container.__class__

        # -- determine container type
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Container type unknown or not supported: %s" % _class)
        _yaml_doc['type'] = _type

        # -- write representation for container types
        if _class is HistContainer:
            _yaml_doc['bin_edges'] = list(map(float, container.bin_edges))
            if container._manual_heights:
                _yaml_doc['bin_heights'] = list(map(float, container.data))  # float64 -> float
                _yaml_doc['underflow'] = float(container.underflow)
                _yaml_doc['overflow'] = float(container.underflow)
            else:
                _yaml_doc['raw_data'] = list(map(float, container.raw_data))  # float64 -> float
        elif _class is IndexedContainer or _class is UnbinnedContainer:
            _yaml_doc['data'] = container.data.tolist()
        elif _class is XYContainer:
            _yaml_doc['x_data'] = container.x.tolist()
            _yaml_doc['y_data'] = container.y.tolist()
        else:
            raise DReprError("Container type unknown or not supported: {}".format(_type))

        # -- write error representation for all container types
        if container.has_errors:
            common_error_tools.write_errors_to_yaml(container, _yaml_doc)

        # write labels for all container types
        if container.label is not None:
            _yaml_doc['label'] = container.label
        _x_axis_label, _y_axis_label = container.axis_labels
        if _x_axis_label is not None:
            _yaml_doc['x_label'] = _x_axis_label
        if _y_axis_label is not None:
            _yaml_doc['y_label'] = _y_axis_label

        return _yaml_doc


class DataContainerYamlReader(YamlReaderMixin, DataContainerDReprBase):
    LOADER = common_error_tools.MatrixYamlLoader

    def __init__(self, input_io_handle):
        super(DataContainerYamlReader, self).__init__(input_io_handle=input_io_handle,
                                                      data_container=None)

    @classmethod
    def _get_required_keywords(cls, yaml_doc, container_class):
        if container_class is HistContainer:
            return []
        if container_class is IndexedContainer or container_class is UnbinnedContainer:
            return ['data']
        if container_class is XYContainer:
            return ['x_data', 'y_data']
        raise YamlReaderException("Unknown container type")

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine container class from type
        _container_type = yaml_doc.pop('type')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_container_type)

        # -- read in representation for container types
        if _class is HistContainer:
            _bin_edges = yaml_doc.pop('bin_edges', None)
            _n_bins = yaml_doc.pop('n_bins', None)
            _bin_range = yaml_doc.pop('bin_range', None)
            if not _bin_edges and not (_n_bins and _bin_range):
                raise YamlReaderException("When reading in a histogram dataset either "
                                          "bin_edges or n_bins and bin_range has to be specified!")
            if _bin_edges:
                _n_bins = len(_bin_edges) - 1
                _bin_range = (_bin_edges[0], _bin_edges[-1])
            _raw_data = yaml_doc.pop('raw_data', None)
            _bin_heights = yaml_doc.pop('bin_heights', None)
            if _raw_data and _bin_heights:
                raise YamlReaderException("When reading in a histogram dataset only one out of "
                                          "raw_data and bin_heights can be specified!")
            _container_obj = HistContainer(n_bins=_n_bins,
                                           bin_range=_bin_range,
                                           bin_edges=_bin_edges,
                                           fill_data=_raw_data)
            if _bin_heights:
                _underflow = yaml_doc.pop('underflow', 0)
                _overflow = yaml_doc.pop('overflow', 0)
                _container_obj.set_bins(
                    bin_heights=_bin_heights, underflow=_underflow, overflow=_overflow)
        elif _class is IndexedContainer:
            _data = yaml_doc.pop('data')
            _container_obj = IndexedContainer(_data)
        elif _class is UnbinnedContainer:
            _data = yaml_doc.pop('data')
            _container_obj = UnbinnedContainer(_data)
        elif _class is XYContainer:
            _x_data = yaml_doc.pop('x_data')
            _y_data = yaml_doc.pop('y_data')
            _container_obj = XYContainer(_x_data, _y_data)
        else:
            raise DReprError("Container type unknown or not supported: {}".format(_container_type))

        # get labels for all container types
        _container_obj.label = yaml_doc.pop('label', None)
        _container_obj.axis_labels = (yaml_doc.pop('x_label', None), yaml_doc.pop('y_label', None))

        # -- process error sources
        _container_obj, yaml_doc = common_error_tools.process_error_sources(
            container_obj=_container_obj, yaml_doc=yaml_doc)

        return _container_obj, yaml_doc


# register the above classes in the module-level dictionary
DataContainerYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
DataContainerYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
