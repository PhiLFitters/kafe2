import sys
import warnings

from .minimizer_base import MinimizerException
from ...config import kc

__all__ = ['get_minimizer']

AVAILABLE_MINIMIZERS = dict()

_MINIMIZER_NAME_ALIASES = dict()

_catch_error_class = ImportError
if sys.version_info >= (3, 6):
    # python version 3.6+ throws a different exception type on import fail...
    _catch_error_class = ModuleNotFoundError

try:
    from .scipy_optimize_minimizer import MinimizerScipyOptimize
    __all__.append('MinimizerScipyOptimize')
    AVAILABLE_MINIMIZERS.update({
        'scipy': MinimizerScipyOptimize,
    })
    _MINIMIZER_NAME_ALIASES['scipy.optimize'] = 'scipy'
except _catch_error_class:
    pass

try:
    from .iminuit_minimizer import MinimizerIMinuit
    __all__.append('MinimizerIMinuit')
    AVAILABLE_MINIMIZERS.update({
        'iminuit': MinimizerIMinuit,
    })
except _catch_error_class:
    pass
except SyntaxError:  # Newer versions of iminuit do not support Python 2.
    pass
except MinimizerException as e:  # kafe2 does not support iminuit>2
    warnings.warn("Problem importing iminuit: {}\nIminuit won't be available".format(e))

try:
    from .root_tminuit_minimizer import MinimizerROOTTMinuit
    __all__.append('MinimizerROOTTMinuit')
    AVAILABLE_MINIMIZERS.update({
        'root.tminuit': MinimizerROOTTMinuit,
    })
    _MINIMIZER_NAME_ALIASES['minuit'] = 'root.tminuit'
    _MINIMIZER_NAME_ALIASES['root'] = 'root.tminuit'
except (_catch_error_class, ImportError):  # Python 2.7 ROOT bindings cause ImportError in Python 3.6
    pass

# raise if no minimizers can be imported
if not AVAILABLE_MINIMIZERS:
    raise RuntimeError("Fatal error: no minimizers found! Please ensure that "
                       "at least one of the following Python packages is installed: "
                       "['scipy', 'iminuit', 'ROOT']")


def get_minimizer(minimizer_spec=None):
    """Creates a MinimizerBase object from a given minimizer name.

    :param minimizer_spec: Name of the minimizer to return.
    :type minimizer_spec: str or None
    :return: MinimizerBase-derived kafe2 minimizer object
    """
    global AVAILABLE_MINIMIZERS
    # for 'None', return the default minimizer
    if minimizer_spec is None:
        # go through the default minimizers in the order specified in config
        _minimizer_specs = kc('core', 'minimizers', 'default_minimizer_list')

        # try every spec until a minimizer is found
        for _minimizer_spec in _minimizer_specs:
            _minimizer_spec = _minimizer_spec.lower()
            _minimizer_spec = _MINIMIZER_NAME_ALIASES.get(_minimizer_spec, _minimizer_spec)
            _minimizer = AVAILABLE_MINIMIZERS.get(_minimizer_spec, None)
            if _minimizer is not None:
                return _minimizer

        raise ValueError(
            "Could not find any minimizer in default list: {}! "
            "Available: {}".format(_minimizer_specs, list(AVAILABLE_MINIMIZERS.keys())))
    _minimizer_spec = minimizer_spec.lower()
    _minimizer_spec = _MINIMIZER_NAME_ALIASES.get(_minimizer_spec, _minimizer_spec)
    _minimizer = AVAILABLE_MINIMIZERS.get(_minimizer_spec, None)
    if _minimizer is not None:
        return _minimizer

    raise ValueError("Unknown minimizer '{}'! "
                     "Available: {}".format(minimizer_spec, list(AVAILABLE_MINIMIZERS.keys())))
