"""This submodule contains objects for handling different data representations. Whole fit objects can be saved and
loaded to and from a YAML file. Support for more file formats might follow in future updates.

:synopsis: This submodule contains objects for handling different data representations.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

# this module-level dict is populated by importing the
# individual modules below
_AVAILABLE_REPRESENTATIONS = dict()

# register different aliases (i.e. file extensions) for file formats
_FILE_FORMAT_ALIAS_RESOLVDICT = dict(yml='yaml')


from ._base import DReprError
from .constraint import *
from .container import *
from .fit import *
from .format import *
from .model import *


def _get_representer(object_type_name, role, file_format):
    _d_roles = _AVAILABLE_REPRESENTATIONS.get(object_type_name, None)
    if _d_roles is None:
        raise DReprError(
            "No representers found for object type '{}'! Available object types "
            "are: {}".format(object_type_name, _AVAILABLE_REPRESENTATIONS.keys()))

    _d_fileformats = _d_roles.get(role, None)
    if _d_fileformats is None:
        raise DReprError(
            "No representers with role '{}' found for object type '{}'! "
            "Available roles are: {}".format(role,
                                             object_type_name,
                                             _d_roles.keys()))

    _file_format_resolve_alias = _FILE_FORMAT_ALIAS_RESOLVDICT.get(file_format, file_format)
    _representer = _d_fileformats.get(_file_format_resolve_alias, None)
    if _representer is None:
        raise DReprError(
            "No representer with role '{}' found for of object type '{}' and file format '{}'! "
            "Available file formats are: {}".format(
                file_format,
                role,
                object_type_name, _d_fileformats.keys()))

    return _representer


# -- convenience functions

def get_reader(object_type_name, file_format):
    return _get_representer(object_type_name=object_type_name,
                            role='reader',
                            file_format=file_format)


def get_writer(object_type_name, file_format):
    return _get_representer(object_type_name=object_type_name,
                            role='writer',
                            file_format=file_format)
