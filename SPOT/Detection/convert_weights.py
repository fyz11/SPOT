# -*- coding: utf-8 -*-
"""
Convert SPOT's Keras .h5 network weights to PyTorch .pt files, without TensorFlow (only h5py + torch).

Supported architectures:

- 'att_unet' / 'unet' segmentation U-Nets (SPOT.Detection.unets.att_unet / unet). Depth, features, input channels and
  labels are inferred from the conv kernel shapes.
- 'yolov3' YOLOv3 detector, see SPOT.Detection.yolov3_torch: experiencor/keras-yolo3 (layers named conv_<idx> /
  bnorm_<idx>) or qqwweee/keras-yolo3 (auto-named conv2d_<n> / batch_normalization_<n>, mapped in creation order).

Mapping: Conv2D kernels HWIO -> OIHW, Conv2DTranspose kernels (H, W, out, in) -> (in, out, H, W), BatchNormalization
gamma/beta/moving_mean/moving_variance -> weight/bias/running_mean/running_var with the Keras epsilon.

Command line: scripts/convert_keras_weights_to_torch.py
"""

import json
import os
import re

import h5py
import numpy as np
import torch


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


def _numbered(layers, prefix):
    """ the layer names <prefix>_<n> sorted by n (Keras auto-names layers in creation order) """
    names = [n for n in layers if re.fullmatch(r'%s_\d+' % prefix, n)]
    return sorted(names, key=lambda n: int(n.rsplit('_', 1)[1]))


def _is_qqwweee_yolov3(layers):
    """ the qqwweee/keras-yolo3 YOLOv3 (auto-named conv2d_<n> / batch_normalization_<n>): 75 convs, 72 with
    batch normalisation, the 3 output convs with a bias """
    convs = _numbered(layers, 'conv2d')
    return (len(convs) == 75 and len(_numbered(layers, 'batch_normalization')) == 72
            and sum(len(layers[n]) == 2 for n in convs) == 3)


def detect_arch(layers):
    if any(re.fullmatch(r'conv_\d+', name) for name in layers) or _is_qqwweee_yolov3(layers):
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


def _qqwweee_to_experiencor_names(layers, layer_configs, convs):
    """ Rename the layers of a qqwweee/keras-yolo3 model to the experiencor 'conv_<idx>' / 'bnorm_<idx>' layout of
    YOLOv3Torch. Both build the same Darknet-53 + 3 heads with the convs created in the same order, so the k-th
    conv2d (and the k-th batch_normalization) maps to the k-th conv (with batch normalisation) of YOLOv3Torch; all kernel
    shapes are checked when copying. """
    q_convs = _numbered(layers, 'conv2d')
    q_bns = _numbered(layers, 'batch_normalization')
    e_convs = sorted(convs)
    e_bns = [i for i in e_convs if convs[i].bnorm is not None]
    if len(q_convs) != len(e_convs) or len(q_bns) != len(e_bns):
        raise ValueError('h5 has %d convs / %d batch norms, YOLOv3Torch %d / %d'
                         % (len(q_convs), len(q_bns), len(e_convs), len(e_bns)))
    renamed, configs = {}, {}
    for q, e in zip(q_convs, e_convs):
        has_bias = len(layers[q]) == 2
        if has_bias != (convs[e].bnorm is None):
            raise ValueError('%s does not match conv_%d (bias / batch normalisation differ)' % (q, e))
        renamed['conv_%d' % e] = layers[q]
    for q, e in zip(q_bns, e_bns):
        renamed['bnorm_%d' % e] = layers[q]
        configs['bnorm_%d' % e] = layer_configs.get(q, {})
    return renamed, configs


def convert_yolov3(layers, layer_configs):
    from SPOT.Detection.yolov3_torch import YOLOv3Torch

    if not any(re.fullmatch(r'conv_\d+', name) for name in layers):
        layers, layer_configs = _qqwweee_to_experiencor_names(layers, layer_configs, YOLOv3Torch().conv_layers())
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
