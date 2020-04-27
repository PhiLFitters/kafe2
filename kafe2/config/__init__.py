"""Handle global kafe2 configuration.
"""

import os
import yaml
import matplotlib as mpl
from copy import deepcopy
from tempfile import NamedTemporaryFile

__all__ = ['ConfigLoader', 'ConfigError', 'ConfigTypeError', 'ConfigLookupError', 'kafe2_rc', 'kc']


class ConfigError(Exception):
    pass


class ConfigLookupError(ConfigError):
    def __init__(self, key_path, problematic_key_index):
        self.message = "Error getting config key for '{}': no node under that path!".format(', '.join(key_path))
        self.message = ("Error getting config key for '{}': "
                        "cannot find config node '{}'!".format(', '.join(key_path), key_path[problematic_key_index]))


class ConfigTypeError(ConfigError):
    def __init__(self, key_path, problematic_key_index):
        self.message = ("Error getting config key for '{}': "
                        "scalar node '{}' encountered inside path!".format(', '.join(key_path), key_path[problematic_key_index]))


class ConfigLoader(yaml.Loader):
    def __init__(self, file_like):
        self._current_dir = os.path.split(file_like.name)[0]
        super(ConfigLoader, self).__init__(file_like)

    # custom directives
    def include(self, node):
        _include_path = self.construct_scalar(node)
        with open(os.path.join(self._current_dir, _include_path)) as _f:
            return yaml.load(_f, ConfigLoader)

ConfigLoader.add_constructor('!include', ConfigLoader.include)

# -- read in default global kafe2 configuration from file
with open(os.path.join(__path__[0], 'kafe2.yaml')) as _f:
    _kc = yaml.load(_f, ConfigLoader)

# make a copy of the default configuration
_default_kc = deepcopy(_kc)

def kc(*keys):
    """Lookup configuration entry by providing the path to it."""
    _dict = _kc

    # if called without args, retreive the entire configuration dict
    if not keys:
        return _kc

    try:
        for _i, _k in enumerate(keys):
            _dict = _dict[_k]
    except KeyError:
        raise ConfigLookupError(keys, _i)
    except TypeError:
        raise ConfigTypeError(keys, _i)

    return _dict


try:
    if mpl.__version__.startswith('2'):
        kafe2_rc = mpl.rc_params_from_file(os.path.join(__path__[0], 'kafe2.matplotlibrc.conf'))
        mpl.rcParams.update(**kafe2_rc)
    elif mpl.__version__.startswith('3'):
        _temp_file = NamedTemporaryFile(delete=False)
        with open(os.path.join(__path__[0], 'kafe2.matplotlibrc.conf')) as _file:
            for _line in _file.readlines():
                if _line.startswith("text.latex.unicode"):
                    continue
                _temp_file.write(_line.encode())
                _temp_file.write('\n'.encode())
        kafe2_rc = mpl.rc_params_from_file(_temp_file.name)
        _temp_file.close()
        os.remove(_temp_file.name)
except AttributeError:
    kafe2_rc = None  # if mpl is only a mock module
