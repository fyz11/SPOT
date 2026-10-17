# -*- coding: utf-8 -*-
"""
Convert SPOT's Keras .h5 network weights to PyTorch .pt files, without TensorFlow (only h5py + torch).

    python scripts/convert_keras_weights_to_torch.py                       # convert the default SPOT models found
    python scripts/convert_keras_weights_to_torch.py path/to/model.h5 ...  # convert specific files

The .pt is written next to each .h5 (same basename). Supported architectures:

- 'att_unet' / 'unet' segmentation U-Nets (SPOT.Detection.unets.att_unet / unet). Depth, features, input channels and
  labels are inferred from the conv kernel shapes.
- 'yolov3' keras-yolo3 YOLOv3 detector (layers named conv_<idx> / bnorm_<idx>), see SPOT.Detection.yolov3_torch.

Mapping: Conv2D kernels HWIO -> OIHW, Conv2DTranspose kernels (H, W, out, in) -> (in, out, H, W), BatchNormalization
gamma/beta/moving_mean/moving_variance -> weight/bias/running_mean/running_var with the Keras epsilon.
"""

import argparse
import json
import os
import re
import sys

import h5py
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

DEFAULT_MODELS = [os.path.join(REPO_DIR, 'models', 'segment_CNN_model', 'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5'),
                  os.path.join(REPO_DIR, 'models', 'detect_CNN_model', 'keras_YOLOv3_organoid_detector2.h5')]


def _attr_str(x):
    return x.decode('utf8') if isinstance(x, bytes) else x


def read_keras_h5(h5file):
    """ Read the per-layer weights of a Keras .h5 (either a full `model.save` file or a `save_weights` file).

    Returns (layers, layer_configs): layers is an ordered {layer_name: [arrays in weight_names order]} for layers
    with weights; layer_configs is {layer_name: config dict} if the model architecture is stored, else {}.
    """
    with h5py.File(h5file, 'r') as f:
        g = f['model_weights'] if 'model_weights' in f else f
        layers = {}
        for name in g.attrs['layer_names']:
            name = _attr_str(name)
            weight_names = [_attr_str(w) for w in g[name].attrs['weight_names']]
            if len(weight_names) > 0:
                layers[name] = [np.asarray(g[name][w]) for w in weight_names]

        layer_configs = {}
        if 'model_config' in f.attrs:
            config = json.loads(_attr_str(f.attrs['model_config']))
            for layer in config['config']['layers']:
                layer_configs[layer['config']['name']] = dict(layer['config'], class_name=layer['class_name'])
    return layers, layer_configs


def conv_kernel_to_torch(kernel):
    """ Keras Conv2D kernel (H, W, in, out) -> torch Conv2d weight (out, in, H, W) """
    return torch.from_numpy(np.ascontiguousarray(kernel.transpose(3, 2, 0, 1)))


def conv_transpose_kernel_to_torch(kernel):
    """ Keras Conv2DTranspose kernel (H, W, out, in) -> torch ConvTranspose2d weight (in, out, H, W).
    Both frameworks define the transposed conv as the adjoint of the same cross-correlation, so no spatial flip. """
    return torch.from_numpy(np.ascontiguousarray(kernel.transpose(3, 2, 0, 1)))


def copy_conv(conv, weights):
    """ copy Keras Conv2D [kernel(, bias)] into a torch nn.Conv2d / nn.ConvTranspose2d """
    kernel = weights[0]
    w = conv_transpose_kernel_to_torch(kernel) if isinstance(conv, torch.nn.ConvTranspose2d) else conv_kernel_to_torch(kernel)
    if tuple(w.shape) != tuple(conv.weight.shape):
        raise ValueError('kernel shape mismatch: keras %s -> torch %s, expected %s' % (kernel.shape, tuple(w.shape), tuple(conv.weight.shape)))
    conv.weight.data.copy_(w)
    if len(weights) > 1:
        if conv.bias is None:
            raise ValueError('keras conv has a bias but torch conv does not')
        conv.bias.data.copy_(torch.from_numpy(weights[1]))
    elif conv.bias is not None:
        raise ValueError('torch conv has a bias but keras conv does not')


def copy_batchnorm(bn, weights, config=None):
    """ copy Keras BatchNormalization weights into torch nn.BatchNorm2d.
    weights are [gamma, beta, moving_mean, moving_variance] (gamma / beta absent if scale / center=False). """
    config = config or {}
    weights = list(weights)
    gamma = weights.pop(0) if config.get('scale', True) else np.ones(bn.num_features, dtype=np.float32)
    beta = weights.pop(0) if config.get('center', True) else np.zeros(bn.num_features, dtype=np.float32)
    mean, var = weights
    bn.weight.data.copy_(torch.from_numpy(gamma))
    bn.bias.data.copy_(torch.from_numpy(beta))
    bn.running_mean.data.copy_(torch.from_numpy(mean))
    bn.running_var.data.copy_(torch.from_numpy(var))
    if 'epsilon' in config:
        bn.eps = float(config['epsilon'])


def detect_arch(layers):
    if any(re.fullmatch(r'conv_\d+', name) for name in layers):
        return 'yolov3'
    kernels = [w[0] for w in layers.values() if w[0].ndim == 4]
    n_conv = len(kernels)
    # only att_unet has 1x1 convs before the final 1x1 output conv (the attention gates)
    has_attention = any(k.shape[:2] == (1, 1) for k in kernels[:-1])
    if has_attention and (n_conv - 3) % 7 == 0:
        return 'att_unet'
    if not has_attention and (n_conv - 3) % 4 == 0:
        return 'unet'
    raise ValueError('could not recognise the architecture of the .h5 (%d conv layers)' % n_conv)


def convert_unet(layers, arch):
    from SPOT.Detection.unets import unet_torch, att_unet_torch

    kernels = [w for w in layers.values() if w[0].ndim == 4]  # conv layers in Keras layer (creation) order
    n_conv = len(kernels)
    depth = (n_conv - 3) // (7 if arch == 'att_unet' else 4)
    config = {'n_label': int(kernels[-1][0].shape[-1]),
              'img_ch': int(kernels[0][0].shape[2]),
              'depth': int(depth),
              'features': int(kernels[0][0].shape[3])}
    model = (att_unet_torch if arch == 'att_unet' else unet_torch)(**config)
    convs = model.keras_conv_layers()
    assert len(convs) == n_conv, (len(convs), n_conv)
    for conv, weights in zip(convs, kernels):
        copy_conv(conv, weights)
    return model, config


def convert_yolov3(layers, layer_configs):
    from SPOT.Detection.yolov3_torch import YOLOv3Torch

    n_out = layers['conv_81'][0].shape[-1]
    config = {'n_classes': int(n_out // 3 - 5)}
    model = YOLOv3Torch(**config)
    convs = model.conv_layers()
    expected = set(['conv_%d' % i for i in convs] + ['bnorm_%d' % i for i, c in convs.items() if c.bnorm is not None])
    missing = expected - set(layers)
    extra = set(layers) - expected
    if missing or extra:
        raise ValueError('h5 layers do not match the keras-yolo3 layout: missing %s, unexpected %s' % (sorted(missing), sorted(extra)))
    for idx, layer in convs.items():
        copy_conv(layer.conv, layers['conv_%d' % idx])
        if layer.bnorm is not None:
            copy_batchnorm(layer.bnorm, layers['bnorm_%d' % idx], layer_configs.get('bnorm_%d' % idx))
    return model, config


def convert(h5file, ptfile=None, arch=None):
    """ convert one Keras .h5 to a .pt holding {'arch', 'config', 'state_dict', 'source'}; returns the .pt path """
    layers, layer_configs = read_keras_h5(h5file)
    arch = arch or detect_arch(layers)
    if arch == 'yolov3':
        model, config = convert_yolov3(layers, layer_configs)
    else:
        model, config = convert_unet(layers, arch)

    ptfile = ptfile or os.path.splitext(h5file)[0] + '.pt'
    torch.save({'arch': arch, 'config': config, 'state_dict': model.state_dict(), 'source': os.path.basename(h5file)}, ptfile)
    return ptfile


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('h5files', nargs='*', help='Keras .h5 files (default: the SPOT models in models/)')
    parser.add_argument('--arch', choices=['att_unet', 'unet', 'yolov3'], default=None, help='override architecture detection')
    args = parser.parse_args()

    h5files = args.h5files or DEFAULT_MODELS
    for h5file in h5files:
        if not os.path.exists(h5file):
            print('skipping %s (not found)' % h5file)
            if 'keras_YOLOv3_organoid_detector2' in h5file:
                print('  download the YOLOv3 weights (see README.md) into models/detect_CNN_model/ and re-run')
            continue
        ptfile = convert(h5file, arch=args.arch)
        print('converted %s -> %s' % (h5file, ptfile))


if __name__ == '__main__':
    main()
