# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:23:37 2021

@author: fyz11
"""

# The tensorflow.keras symbols used by the Keras networks below are imported lazily by _import_keras() so that this
# module (and the PyTorch variants unet_torch / att_unet_torch) can be imported without TensorFlow installed.
def _import_keras():
    global Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, add, multiply
    global concatenate, Dropout, Lambda, Model, Adam, SGD, K
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input,         add, multiply
    from tensorflow.keras.layers import concatenate, Activation, Dropout, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers import SGD
    # from tensorflow.keras.layers.core import Lambda
    import tensorflow.keras.backend as K


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None


def up_and_concate(down_layer, layer, data_format='channels_first'):
    _import_keras()
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate


def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    _import_keras()
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    _import_keras()
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x


def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

              padding='same', data_format='channels_first'):
    _import_keras()
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format='channels_first'):
    _import_keras()
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer

########################################################################################################
# Define the neural network
def unet(img_w, img_h, n_label, img_ch=1, depth=4, features=64, data_format='channels_first'):
    _import_keras()
    inputs = Input((img_ch, img_w, img_h))
    x = inputs
    # depth = 4
    # features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format= data_format)(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        # attention_up_and_concate(x,[skips[i])
        x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        x = concatenate([skips[i], x], axis=1)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
#Attention U-Net
def att_unet(img_w, img_h, n_label, img_ch=1, depth=4, features=64, data_format='channels_first'):
    _import_keras()
    
    if data_format == 'channels_first':
        inputs = Input((img_ch, img_w, img_h))
    if data_format =='channels_last':
        inputs = Input((img_w, img_h, img_ch))
        
    x = inputs
    # depth = depth
    # features = features
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
# PyTorch variants of unet / att_unet. These reproduce the Keras networks layer by layer (always channels_first, i.e.
# NCHW tensors) so that weights trained in Keras can be converted with scripts/convert_keras_weights_to_torch.py.
# keras_conv_layers() lists the convolutions in the order the Keras functions above create them, which is the order
# of the Conv2D layers in the Keras .h5 file.
def _require_torch():
    if torch is None:
        raise ImportError('PyTorch is required for unet_torch / att_unet_torch. Install it with `pip install torch`.')


if torch is not None:

    class _Conv2dSame(nn.Conv2d):
        """ Conv2D(..., padding='same') with odd kernel and stride 1, as used throughout the Keras U-Nets """
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


    class AttentionBlock2dTorch(nn.Module):
        """ PyTorch version of attention_block_2d """
        def __init__(self, x_channels, g_channels, inter_channel):
            super().__init__()
            self.theta_x = _Conv2dSame(x_channels, inter_channel, 1)
            self.phi_g = _Conv2dSame(g_channels, inter_channel, 1)
            self.psi_f = _Conv2dSame(inter_channel, 1, 1)

        def forward(self, x, g):
            f = F.relu(self.theta_x(x) + self.phi_g(g))
            rate = torch.sigmoid(self.psi_f(f))
            return x * rate


    class UNetTorch(nn.Module):
        """ PyTorch version of unet (attention=False) and att_unet (attention=True).

        Input is (N, img_ch, H, W), output is the (N, n_label, H, W) sigmoid probabilities. H and W must be divisible
        by 2**depth. Dropout layers of the Keras model are inactive at inference and are omitted.
        """
        def __init__(self, n_label, img_ch=1, depth=4, features=64, attention=False):
            super().__init__()
            self.n_label = n_label
            self.img_ch = img_ch
            self.depth = depth
            self.features = features
            self.attention = attention

            self.down = nn.ModuleList()
            in_ch = img_ch
            skip_ch = []
            for i in range(depth):
                self.down.append(nn.ModuleList([_Conv2dSame(in_ch, features, 3), _Conv2dSame(features, features, 3)]))
                skip_ch.append(features)
                in_ch = features
                features = features * 2

            self.bottom = nn.ModuleList([_Conv2dSame(in_ch, features, 3), _Conv2dSame(features, features, 3)])
            in_ch = features

            self.att = nn.ModuleList()
            self.up = nn.ModuleList()
            for i in reversed(range(depth)):
                features = features // 2
                if attention:
                    # attention_up_and_concate uses inter_channel = in_channel // 4 of the upsampled layer
                    self.att.append(AttentionBlock2dTorch(skip_ch[i], in_ch, in_ch // 4))
                self.up.append(nn.ModuleList([_Conv2dSame(in_ch + skip_ch[i], features, 3), _Conv2dSame(features, features, 3)]))
                in_ch = features

            self.final = _Conv2dSame(in_ch, n_label, 1)

        def keras_conv_layers(self):
            """ conv layers in the creation order of the Keras unet / att_unet functions """
            convs = []
            for block in self.down:
                convs += list(block)
            convs += list(self.bottom)
            for j, block in enumerate(self.up):
                if self.attention:
                    convs += [self.att[j].theta_x, self.att[j].phi_g, self.att[j].psi_f]
                convs += list(block)
            convs.append(self.final)
            return convs

        def forward(self, x):
            skips = []
            for conv1, conv2 in self.down:
                x = F.relu(conv2(F.relu(conv1(x))))
                skips.append(x)
                x = F.max_pool2d(x, 2)

            conv1, conv2 = self.bottom
            x = F.relu(conv2(F.relu(conv1(x))))

            for j, (conv1, conv2) in enumerate(self.up):
                skip = skips[self.depth - 1 - j]
                x = F.interpolate(x, scale_factor=2, mode='nearest')
                if self.attention:
                    # att_unet concatenates [upsampled, attention gated skip]
                    x = torch.cat([x, self.att[j](skip, x)], dim=1)
                else:
                    # unet concatenates [skip, upsampled]
                    x = torch.cat([skip, x], dim=1)
                x = F.relu(conv2(F.relu(conv1(x))))

            return torch.sigmoid(self.final(x))


def unet_torch(n_label, img_ch=1, depth=4, features=64):
    """ PyTorch variant of unet (channels_first). Image size is not fixed but must be divisible by 2**depth. """
    _require_torch()
    return UNetTorch(n_label, img_ch=img_ch, depth=depth, features=features, attention=False)


def att_unet_torch(n_label, img_ch=1, depth=4, features=64):
    """ PyTorch variant of att_unet (channels_first). Image size is not fixed but must be divisible by 2**depth. """
    _require_torch()
    return UNetTorch(n_label, img_ch=img_ch, depth=depth, features=features, attention=True)


def load_unet_torch(ptfile, device='cpu'):
    """ Load a .pt U-Net written by scripts/convert_keras_weights_to_torch.py and return it in eval mode. """
    _require_torch()
    ckpt = torch.load(ptfile, map_location='cpu')
    builder = {'unet': unet_torch, 'att_unet': att_unet_torch}[ckpt['arch']]
    model = builder(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()


########################################################################################################
#Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def r2_unet(img_w, img_h, n_label, img_ch=1, depth=4, features=64, data_format='channels_first'):
    _import_keras()
    inputs = Input((img_ch, img_w, img_h))
    x = inputs
    # depth = 4
    # features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
#Attention R2U-Net
def att_r2_unet(img_w, img_h, n_label, img_ch=1, depth=4, features=64, data_format='channels_first'):
    _import_keras()
    inputs = Input((img_ch, img_w, img_h))
    x = inputs
    # depth = 4
    # features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    return model


