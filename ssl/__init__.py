from ssl.base import BaseModel
from ssl.simclr import SimCLR
from ssl.moco import MoCo
from ssl.byol import BYOL
from ssl.simsiam import SimSiam
from ssl.swav import SwAV
from ssl.dino import DINO
from ssl.mae import MAE

_method_class_map = {
    'base': BaseModel,
    'simclr': SimCLR,
    'moco': MoCo,
    'byol': BYOL,
    'simsiam': SimSiam,
    'swav': SwAV,
    'dino': DINO,
    'mae': MAE
}


def get_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
