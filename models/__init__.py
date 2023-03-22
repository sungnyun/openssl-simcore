from .resnet import ResNet10, ResNet18, ResNet50, ResNet101
from .resnext import resnext50, resnext101
from .efficientnet import EfficientNet_B0, EfficientNet_B1, EfficientNet_B2
from .dino_vit import dino_vit_tiny, dino_vit_small, dino_vit_base
from .mae_vit import mae_vit_base_patch16_dec512d8b, mae_vit_large_patch16_dec512d8b, mae_vit_huge_patch14_dec512d8b
# from .timm_vit import vit_small_patch16_224, vit_small_patch32_224, vit_base_patch8_224, vit_base_patch16_224, vit_base_patch32_224, vit_large_patch16_224, vit_large_patch32_224
# from .timm_vit import vit_small_patch8_224_dino, vit_small_patch16_224_dino, vit_base_patch8_224_dino, vit_base_patch16_224_dino

_backbone_class_map = {
    'resnet10': ResNet10,
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnext50': resnext50,
    'resnext101': resnext101,
    'efficientnet_b0': EfficientNet_B0,
    'efficientnet_b1': EfficientNet_B1,
    'efficientnet_b2': EfficientNet_B2,
    'dino_vit_t16': dino_vit_tiny,
    'dino_vit_s16': dino_vit_small,
    'dino_vit_b16': dino_vit_base,
    'mae_vit_base': mae_vit_base_patch16_dec512d8b,
    'mae_vit_large': mae_vit_large_patch16_dec512d8b,
    'mae_vit_huge': mae_vit_huge_patch14_dec512d8b
}


def get_backbone_class(key):
    if key in _backbone_class_map:
        return _backbone_class_map[key]
    else:
        raise ValueError('Invalid backbone: {}'.format(key))