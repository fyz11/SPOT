# -*- coding: utf-8 -*-
"""
PyTorch variant of the Keras YOLOv3 (Darknet-53) detector loaded by
SPOT.Detection.detection.load_and_run_YOLOv3_weights_keras_detection_model_RGB.

The layer layout follows keras-yolo3 (experiencor, `make_yolov3_model` / `create_yolov3_model`) which the decoding
helpers in detection.py (_decode_netout, _correct_yolo_boxes, _do_nms) come from: every convolution is named
'conv_<idx>' with an optional 'bnorm_<idx>' (BatchNormalization, epsilon=1e-3) and LeakyReLU(0.1); stride-2
convolutions are preceded by ZeroPadding2D(((1,0),(1,0))) and use 'valid' padding. Keeping the same <idx> as module
names lets scripts/convert_keras_weights_to_torch.py copy the Keras weights by name.

The network outputs the 3 heads (strides 32, 16, 8) as NCHW tensors, in the same order as the Keras model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _c(idx, filters, kernel, stride=1, bnorm=True, leaky=True):
    return {'layer_idx': idx, 'filter': filters, 'kernel': kernel, 'stride': stride, 'bnorm': bnorm, 'leaky': leaky}


class YoloConv(nn.Module):
    """ conv_<idx> [+ bnorm_<idx>] [+ leaky_<idx>] """
    def __init__(self, in_channels, spec, bn_eps=1e-3):
        super().__init__()
        self.stride = spec['stride']
        self.leaky = spec['leaky']
        kernel = spec['kernel']
        # stride 1 uses Keras padding='same' (odd kernel -> symmetric), stride 2 pads top/left by 1 then 'valid'.
        self.conv = nn.Conv2d(in_channels, spec['filter'], kernel, stride=self.stride,
                              padding=kernel // 2 if self.stride == 1 else 0, bias=not spec['bnorm'])
        self.bnorm = nn.BatchNorm2d(spec['filter'], eps=bn_eps) if spec['bnorm'] else None

    def forward(self, x):
        if self.stride > 1:
            x = F.pad(x, (1, 0, 1, 0))
        x = self.conv(x)
        if self.bnorm is not None:
            x = self.bnorm(x)
        if self.leaky:
            x = F.leaky_relu(x, 0.1)
        return x


class YoloConvBlock(nn.Module):
    """ PyTorch version of keras-yolo3 `_conv_block`: the residual skip is taken before the second-to-last conv. """
    def __init__(self, in_channels, convs, skip=True, bn_eps=1e-3):
        super().__init__()
        self.skip = skip
        self.layers = nn.ModuleDict()
        for spec in convs:
            self.layers[str(spec['layer_idx'])] = YoloConv(in_channels, spec, bn_eps=bn_eps)
            in_channels = spec['filter']
        self.out_channels = in_channels

    def forward(self, x):
        n = len(self.layers)
        skip_connection = None
        for count, layer in enumerate(self.layers.values()):
            if count == n - 2 and self.skip:
                skip_connection = x
            x = layer(x)
        return skip_connection + x if self.skip else x


class YOLOv3Torch(nn.Module):
    """ Darknet-53 YOLOv3 with 3 detection heads of 3 * (5 + n_classes) channels each. """
    def __init__(self, n_classes=1, n_anchors_per_scale=3, bn_eps=1e-3):
        super().__init__()
        n_out = n_anchors_per_scale * (5 + n_classes)
        self.n_classes = n_classes
        self.blocks = nn.ModuleDict()

        def block(name, in_ch, convs, skip=True):
            self.blocks[name] = YoloConvBlock(in_ch, convs, skip=skip, bn_eps=bn_eps)
            return self.blocks[name].out_channels

        # Darknet-53 backbone
        ch = block('b0', 3, [_c(0, 32, 3), _c(1, 64, 3, 2), _c(2, 32, 1), _c(3, 64, 3)])
        ch = block('b5', ch, [_c(5, 128, 3, 2), _c(6, 64, 1), _c(7, 128, 3)])
        ch = block('b9', ch, [_c(9, 64, 1), _c(10, 128, 3)])
        ch = block('b12', ch, [_c(12, 256, 3, 2), _c(13, 128, 1), _c(14, 256, 3)])
        for i in range(7):
            ch = block('b%d' % (16 + i * 3), ch, [_c(16 + i * 3, 128, 1), _c(17 + i * 3, 256, 3)])
        ch_36 = ch
        ch = block('b37', ch, [_c(37, 512, 3, 2), _c(38, 256, 1), _c(39, 512, 3)])
        for i in range(7):
            ch = block('b%d' % (41 + i * 3), ch, [_c(41 + i * 3, 256, 1), _c(42 + i * 3, 512, 3)])
        ch_61 = ch
        ch = block('b62', ch, [_c(62, 1024, 3, 2), _c(63, 512, 1), _c(64, 1024, 3)])
        for i in range(3):
            ch = block('b%d' % (66 + i * 3), ch, [_c(66 + i * 3, 512, 1), _c(67 + i * 3, 1024, 3)])

        # head at stride 32
        ch = block('b75', ch, [_c(75, 512, 1), _c(76, 1024, 3), _c(77, 512, 1), _c(78, 1024, 3), _c(79, 512, 1)], skip=False)
        block('b80', ch, [_c(80, 1024, 3), _c(81, n_out, 1, bnorm=False, leaky=False)], skip=False)
        # head at stride 16
        ch = block('b84', ch, [_c(84, 256, 1)], skip=False)
        ch = block('b87', ch + ch_61, [_c(87, 256, 1), _c(88, 512, 3), _c(89, 256, 1), _c(90, 512, 3), _c(91, 256, 1)], skip=False)
        block('b92', ch, [_c(92, 512, 3), _c(93, n_out, 1, bnorm=False, leaky=False)], skip=False)
        # head at stride 8
        ch = block('b96', ch, [_c(96, 128, 1)], skip=False)
        block('b99', ch + ch_36, [_c(99, 128, 1), _c(100, 256, 3), _c(101, 128, 1), _c(102, 256, 3),
                                  _c(103, 128, 1), _c(104, 256, 3), _c(105, n_out, 1, bnorm=False, leaky=False)], skip=False)

    def conv_layers(self):
        """ {layer_idx: YoloConv} used to map the Keras 'conv_<idx>' / 'bnorm_<idx>' weights """
        return {int(idx): layer for blk in self.blocks.values() for idx, layer in blk.layers.items()}

    def forward(self, x):
        b = self.blocks
        x = b['b0'](x)
        x = b['b5'](x)
        x = b['b9'](x)
        x = b['b12'](x)
        for i in range(7):
            x = b['b%d' % (16 + i * 3)](x)
        skip_36 = x
        x = b['b37'](x)
        for i in range(7):
            x = b['b%d' % (41 + i * 3)](x)
        skip_61 = x
        x = b['b62'](x)
        for i in range(3):
            x = b['b%d' % (66 + i * 3)](x)

        x = b['b75'](x)
        yolo_82 = b['b80'](x)

        x = b['b84'](x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip_61], dim=1)
        x = b['b87'](x)
        yolo_94 = b['b92'](x)

        x = b['b96'](x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip_36], dim=1)
        yolo_106 = b['b99'](x)

        return [yolo_82, yolo_94, yolo_106]


def load_yolov3_torch(ptfile, device='cpu'):
    """ Load a .pt YOLOv3 written by scripts/convert_keras_weights_to_torch.py and return it in eval mode. """
    ckpt = torch.load(ptfile, map_location='cpu')
    model = YOLOv3Torch(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()
