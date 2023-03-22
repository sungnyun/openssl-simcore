from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, pretrained=''):
    if backbone == 'resnet34':
#         return resnet.ResNet101(output_stride, BatchNorm)
        return resnet.ResNet34(pretrained=pretrained)
    elif backbone == 'resnet50':
#         return resnet.ResNet101(output_stride, BatchNorm)
        return resnet.ResNet50(pretrained=pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
