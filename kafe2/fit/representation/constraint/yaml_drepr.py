from .._base import DReprError
from .._yaml_base import YamlWriterMixin, YamlReaderMixin
from ._base import ConstraintDReprBase
from .. import _AVAILABLE_REPRESENTATIONS
from ....core.constraint import GaussianSimpleParameterConstraint, GaussianMatrixParameterConstraint

__all__ = ['ConstraintYamlWriter', 'ConstraintYamlReader']


class ConstraintYamlWriter(YamlWriterMixin, ConstraintDReprBase):

    def __init__(self, constraint, output_io_handle):
        super(ConstraintYamlWriter, self).__init__(
            output_io_handle=output_io_handle,
            constraint=constraint)

    @classmethod
    def _make_representation(cls, constraint):
        _yaml_doc = dict()

        # -- determine constraint type
        _class = constraint.__class__
        _type = cls._CLASS_TO_OBJECT_TYPE_NAME.get(_class, None)
        if _type is None:
            raise DReprError("Constraint type unknown or not supported: %s" % constraint.__class__)
        _yaml_doc['type'] = _type

        if _class is GaussianSimpleParameterConstraint:
            _yaml_doc['index'] = constraint.index
            _yaml_doc['value'] = constraint.value
            _yaml_doc['uncertainty'] = constraint.uncertainty
            _yaml_doc['relative'] = constraint.relative
        elif _class is GaussianMatrixParameterConstraint:
            _yaml_doc['indices'] = constraint.indices.tolist()
            _yaml_doc['values'] = constraint.values.tolist()
            if constraint.relative:
                if constraint.matrix_type == 'cov':
                    _yaml_doc['matrix'] = constraint.cov_mat_rel.tolist()
                else:
                    _yaml_doc['matrix'] = constraint.cor_mat.tolist()
                    _yaml_doc['uncertainties'] = constraint.uncertainties_rel.tolist()
            else:
                if constraint.matrix_type == 'cov':
                    _yaml_doc['matrix'] = constraint.cov_mat.tolist()
                else:
                    _yaml_doc['matrix'] = constraint.cor_mat.tolist()
                    _yaml_doc['uncertainties'] = constraint.uncertainties.tolist()
            _yaml_doc['relative'] = constraint.relative
            _yaml_doc['matrix_type'] = constraint.matrix_type
        else:
            raise DReprError('Unknown constraint class: %s' % _class)

        return _yaml_doc


class ConstraintYamlReader(YamlReaderMixin, ConstraintDReprBase):

    def __init__(self, input_io_handle):
        super(ConstraintYamlReader, self).__init__(
            input_io_handle=input_io_handle,
            constraint=None)

    @classmethod
    def _get_required_keywords(cls, yaml_doc, constraint_class):
        if constraint_class is GaussianSimpleParameterConstraint:
            return ['index', 'value', 'uncertainty']
        if constraint_class is GaussianMatrixParameterConstraint:
            return ['indices', 'values', 'matrix']
        raise DReprError('Unknown constraint class: %s' % constraint_class)

    @classmethod
    def _modify_yaml_doc(cls, yaml_doc, kafe_object_class, parameter_names=None, **kwargs):
        if kafe_object_class is GaussianSimpleParameterConstraint:
            _par_name = yaml_doc.pop('name', None)
            if _par_name:
                yaml_doc['index'] = parameter_names.index(_par_name)
        elif kafe_object_class is GaussianMatrixParameterConstraint:
            _par_names = yaml_doc.pop('names', None)
            if _par_names:
                yaml_doc['indices'] = [parameter_names.index(_par_name) for _par_name in _par_names]
        return yaml_doc

    @classmethod
    def _convert_yaml_doc_to_object(cls, yaml_doc):
        # -- determine constraint class from type
        _constraint_type = yaml_doc.pop('type', 'simple')
        _class = cls._OBJECT_TYPE_NAME_TO_CLASS.get(_constraint_type, None)

        if _class is GaussianSimpleParameterConstraint:
            _constraint_object = GaussianSimpleParameterConstraint(
                index=yaml_doc.pop('index'),
                value=yaml_doc.pop('value'),
                uncertainty=yaml_doc.pop('uncertainty'),
                relative=yaml_doc.pop('relative', False)
            )
        elif _class is GaussianMatrixParameterConstraint:
            _constraint_object = GaussianMatrixParameterConstraint(
                indices=yaml_doc.pop('indices'),
                values=yaml_doc.pop('values'),
                matrix=yaml_doc.pop('matrix'),
                matrix_type=yaml_doc.pop('matrix_type', 'cov'),
                uncertainties=yaml_doc.pop('uncertainties', None),
                relative=yaml_doc.pop('relative', False)
            )
        else:
            raise DReprError('Unknown constraint class: %s' % _class)
        return _constraint_object, yaml_doc


# register the above classes in the module-level dictionary
ConstraintYamlReader._register_class(_AVAILABLE_REPRESENTATIONS)
ConstraintYamlWriter._register_class(_AVAILABLE_REPRESENTATIONS)
