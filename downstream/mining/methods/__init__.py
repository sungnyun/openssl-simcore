from .hard_sampling import HardSampling
from .explicit_hard_sampling import ExpHardSampling
from .explicit_hard_sampling_memory import ExpHardSamplingMemory

_method_class_map = {
    'sampling': HardSampling,
    'explicit_sampling': ExpHardSampling,
    'explicit_sampling_memory': ExpHardSamplingMemory
}


def get_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
