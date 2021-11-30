import tensorflow as tf
import numpy as np
from numpy import *


def var_conv_kernel(ch_in, ch_out, k_conv=None, initialiser=None, name='W'):
    with tf.compat.v1.variable_scope(name):
        if k_conv is None:
            k_conv = [3, 3, 3]
        if initialiser is None:
            initialiser = tf.keras.initializers.glorot_normal()
        return tf.compat.v1.get_variable(name, shape=k_conv + [ch_in] + [ch_out],
                               initializer=initialiser)

def var_bias(b_shape, initialiser=None, name='b'):
    with tf.compat.v1.variable_scope(name):
        if initialiser is None:
            initialiser = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape=b_shape, initializer=initialiser)


def var_projection(shape_, initialiser=None, name='P'):
    with tf.variable_scope(name):
        if initialiser is None:
            initialiser = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape=shape_, initializer=initialiser)



def conv3_block(input_, ch_in, ch_out, k_conv=None, strides=None, name='conv3_block'):
    if strides is None:
        strides = [1, 1, 1, 1, 1]
    with tf.compat.v1.variable_scope(name):
        w = var_conv_kernel(ch_in, ch_out, k_conv)

        return tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.nn.conv3d(input_, w, strides, "SAME")))


def conv3D(input_, ch_in, ch_out, k_conv=None, strides=None, name='conv3_block'):
    if strides is None:
        strides = [1, 1, 1, 1, 1]
    with tf.compat.v1.variable_scope(name):
        w = var_conv_kernel(ch_in, ch_out, k_conv)

        return tf.nn.conv3d(input_, w, strides, "SAME")



def deconv3_block(input_, ch_in, ch_out, shape_out, strides, name='deconv3_block'):
    with tf.compat.v1.variable_scope(name):
        w = var_conv_kernel(ch_in, ch_out)
        return tf.nn.relu(
            tf.compat.v1.layers.batch_normalization(tf.nn.conv3d_transpose(input_, w, shape_out, strides, "SAME")))


def downsample_resnet_block(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, name='down_resnet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    with tf.compat.v1.variable_scope(name):
        h0 = conv3_block(input_, ch_in, ch_out, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_out, ch_out, name='WR1')
        wr2 = var_conv_kernel(ch_out, ch_out)

        r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(r2, k_pool, strides2, padding="SAME")
        else:
            w1 = var_conv_kernel(ch_out, ch_out)
            h1 = conv3_block(r2, w1, strides2, name='W1')
        return h1, h0


def upsample_resnet_block(input_, input_skip, ch_in, ch_out, use_additive_upsampling=True, name='up_resnet_block'):
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_skip.shape.as_list()
    with tf.variable_scope(name):
        h0 = deconv3_block(input_, ch_out, ch_in, size_out, strides2)
        if use_additive_upsampling:
            h0 += additive_up_sampling(input_, size_out[1:4])
        r1 = h0 + input_skip
        r2 = conv3_block(h0, ch_out, ch_out)
        wr2 = var_conv_kernel(ch_out, ch_out)
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r2, wr2, strides1, "SAME")) + r1)
        return h1


def ddf_summand(input_, ch_in, shape_out, name='ddf_summand'):
    strides1 = [1, 1, 1, 1, 1]
    initial_bias_local = 0.0
    initial_std_local = 0.0
    with tf.variable_scope(name):

        w = var_conv_kernel(ch_in, 3, initialiser=tf.random_normal_initializer(0, initial_std_local))
        b = var_bias([3], initialiser=tf.constant_initializer(initial_bias_local))
        if input_.get_shape() == shape_out:
            return tf.nn.conv3d(input_, w, strides1, "SAME") + b
        # else:
        #     return resize_volume(tf.nn.conv3d(input_, w, strides1, "SAME") + b, shape_out)


# layers
def fully_connected(input_, length_out, initial_bias_global=0.0, name='fully_connected'):
    initial_std_global = 0.0
    input_size = input_.shape.as_list()
    with tf.variable_scope(name):

        w = var_projection([input_size[1] * input_size[2] * input_size[3] * input_size[4], length_out],
                           initialiser=tf.random_normal_initializer(0, initial_std_global))

        b = var_bias([1, length_out], initialiser=tf.constant_initializer(initial_bias_global))
        return tf.matmul(tf.reshape(input_, [input_size[0], -1]), w) + b


def affine_layer_0(input_, length_out, initial_bias_global=0.0, name='fully_connected'):
    initial_std_global = 0.0
    input_size = input_.shape.as_list()
    with tf.variable_scope(name):

        w = var_projection([input_size[1] * input_size[2] * input_size[3] * input_size[4], length_out],
                           initialiser=tf.random_normal_initializer(0, initial_std_global))

        b = var_bias([1, length_out], initialiser=tf.constant_initializer(initial_bias_global))
        return tf.clip_by_value(tf.matmul(tf.reshape(input_, [input_size[0], -1]), w), -0.1, 0.1) + b


def affine_layer_1(input_, length_out, initial_bias_global=0.0, name='fully_connected'):
    initial_std_global = 0.0
    input_size = input_.shape.as_list()
    with tf.variable_scope(name):

        w = var_projection([input_size[1] * input_size[2] * input_size[3] * input_size[4], length_out],
                           initialiser=tf.random_normal_initializer(0, initial_std_global))

        b = var_bias([1, length_out], initialiser=tf.constant_initializer(initial_bias_global))
        return tf.clip_by_value(tf.matmul(tf.reshape(input_, [input_size[0], -1]), w), -0.1, 0.1) + b


def affine_layer_2(input_, length_out, initial_bias_global=0.0, name='fully_connected'):
    initial_std_global = 0.0
    input_size = input_.shape.as_list()
    with tf.variable_scope(name):

        w = var_projection([input_size[1] * input_size[2] * input_size[3] * input_size[4], length_out],
                           initialiser=tf.random_normal_initializer(0, initial_std_global))

        b = var_bias([1, length_out], initialiser=tf.constant_initializer(initial_bias_global))
        return tf.clip_by_value(tf.matmul(tf.reshape(input_, [input_size[0], -1]), w), -0.1, 0.1) + b


def additive_up_sampling(input_, size, stride=2, name='additive_upsampling'):
    with tf.variable_scope(name):
        return tf.reduce_sum(tf.stack(tf.split(input_, stride, axis=4), axis=5), axis=5)


# def resize_volume(image, size, method=0, name='resize_volume'):
#     # size is [depth, height width]
#     # image is Tensor with shape [batch, depth, height, width, channels]
#     shape = image.get_shape().as_list()
#     with tf.variable_scope(name):
#         reshaped2d = tf.reshape(image, [-1, shape[2], shape[3], shape[4]])
#         resized2d = tf.image.resize(reshaped2d, [size[1], size[2]], method)
#         reshaped2d = tf.reshape(resized2d, [shape[0], shape[1], size[1], size[2], shape[4]])
#         permuted = tf.transpose(reshaped2d, [0, 3, 2, 1, 4])
#         reshaped2db = tf.reshape(permuted, [-1, size[1], shape[1], shape[4]])
#         resized2db = tf.image.resize(reshaped2db, [size[1], size[0]], method)
#         reshaped2db = tf.reshape(resized2db, [shape[0], size[2], size[1], size[0], shape[4]])
#         return tf.transpose(reshaped2db, [0, 3, 2, 1, 4])


def downsample_ResUnet_block_1(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    with tf.variable_scope(name):
        h0 = conv3_block(input_, ch_in, ch_out, k_conv0, name='W0')
        h01 = conv3_block(h0, ch_out, ch_out, name='W01')
        # r1 = conv3_block(h01, ch_out, ch_out * 2, name='WR1')
        # r1 = tf.nn.conv3d(h01, ch_out, ch_out * 2, name='WR1')
        wr2 = var_conv_kernel(ch_out, ch_out)

        r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h01, wr2, strides1, "SAME")) + h0)
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(r2, k_pool, strides2, padding="SAME")
        else:
            w1 = var_conv_kernel(ch_out, ch_out)
            h1 = conv3_block(r2, w1, strides2, name='W1')
        return h1, r2


def sme_subnet(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, name='encode_fixed_subnetwork'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides2 = [1, 2, 2, 2, 1]
    with tf.compat.v1.variable_scope(name):
        h01 = conv3_block(input_, ch_in, ch_out, k_conv0, name='W01')
        h02 = conv3_block(h01, ch_out, ch_out, name='W02')
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(h02, k_pool, strides2, padding="SAME")
        else:
            # size = tf.shape(h02, name='sap', out_type='int32')
            # w1 = var_conv_kernel(ch_out, ch_out)
            # h1 = conv3_block(h02, w1, strides2, name='W11')
            k_pool = [1, 2, 2, 2, 1]
            # h1 = tf.nn.max_pool3d(h02, k_pool, strides2, padding="SAME")
            size = h02.get_shape().as_list()
            size[0] = 1
            size[4] = 1
            # input1 = tf.pad(input1, [[0, 0], [0, 16], [0, 0], [0, 8], [0, 0]])
            # h1 = tf.nn.avg_pool3d(h02, ksize=size, strides=[1, 1, 1, 1, 1], padding="VALID")
            h1 = tf.pad(h02, [[0, 0], [0, 4], [0, 0], [0, 2], [0, 0]])
        return h1, h02

def sme_subnet_1(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, name='encode_fixed_subnetwork'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides2 = [1, 2, 2, 2, 1]
    with tf.variable_scope(name):
        h01 = conv3_block(input_, ch_in, ch_out, k_conv0, name='W01')
        h02 = conv3_block(h01, ch_out, ch_out, name='W02')
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(h02, k_pool, strides2, padding="SAME")
        else:
            # size = tf.shape(h02, name='sap', out_type='int32')
            # w1 = var_conv_kernel(ch_out, ch_out)
            # h1 = conv3_block(h02, w1, strides2, name='W11')
            size = h02.get_shape().as_list()
            size[0] = 1
            size[4] = 1
            # input1 = tf.pad(input1, [[0, 0], [0, 16], [0, 0], [0, 8], [0, 0]])
            # h1 = tf.nn.avg_pool3d(h02, ksize=size, strides=[1, 1, 1, 1, 1], padding="VALID")
            h1 = tf.pad(h02, [[0, 0], [0, 4], [0, 0], [0, 2], [0, 0]])
        return h1, h02

def downsample_ResUnet_block(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    with tf.variable_scope(name):

        h0 = conv3_block(input_, ch_in, ch_out, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_out, ch_out, name='WR1')
        wr2 = var_conv_kernel(ch_out, ch_out)

        r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(r1, k_pool, strides2, padding="SAME")
        else:
            w1 = var_conv_kernel(ch_out, ch_out)
            h1 = conv3_block(r1, w1, strides2, name='W1')
        return h1, r2


def upsample_ResUnet_block(input_1, input_2, ch_in, ch_out, k_conv0=None, name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_2.shape.as_list()
    size_out[4] = size_out[4] * 2
    with tf.variable_scope(name):
        h0 = conv3_block(input_1, ch_in + ch_out, ch_out, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_out, ch_out, name='WR1')
        wr2 = var_conv_kernel(ch_out, ch_out)

        r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)

        r2 = deconv3_block(r2, ch_out, ch_out, size_out, strides2)
        # h0 = deconv3_block(input_, ch_out, ch_in, size_out, strides2)
        h1 = tf.concat([r2, input_2], axis=-1)
        return h1


def upsample_ResUnet_block_1(input_1, input_2, ch_in, ch_out, k_conv0=None, name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_2.shape.as_list()
    size_out[4] = size_out[4] * 2
    with tf.variable_scope(name):
        h0 = conv3_block(input_1, ch_in, ch_out, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_out, ch_out, name='WR1')

        wr2 = var_conv_kernel(ch_out, ch_out)

        r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)
        r2 = deconv3_block(r2, ch_out, ch_out, size_out, strides2)
        # h0 = deconv3_block(input_, ch_out, ch_in, size_out, strides2)
        h1 = tf.concat([r2, input_2], axis=-1)
        return h1


def downsample_Unet_block_1(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, name='down_Unet_block1'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    # strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    with tf.compat.v1.variable_scope(name):
        h0 = conv3_block(input_, ch_in, ch_out, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_out, ch_out * 2, name='WR1')
        # wr2 = var_conv_kernel(ch_out, ch_out)

        # r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(r1, k_pool, strides2, padding="SAME")
        else:
            w1 = var_conv_kernel(ch_out, ch_out)
            h1 = conv3_block(r1, w1, strides2, name='W1')
        return h1, r1


def downsample_AttentionUnet_block_1(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, name='down_Unet_block1'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    # strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    with tf.variable_scope(name):
        h0 = conv3_block(input_, ch_in, ch_out, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_out, ch_out * 2, name='WR1')
        # wr2 = var_conv_kernel(ch_out, ch_out)

        # r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(r1, k_pool, strides2, padding="VALID")
        else:
            w1 = var_conv_kernel(ch_out, ch_out)
            h1 = conv3_block(r1, w1, strides2, name='W1')
        return h1, r1


def downsample_Unet_block(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    # strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    with tf.compat.v1.variable_scope(name):

        h0 = conv3_block(input_, ch_in, ch_in, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_in, ch_out, name='WR1')
        # wr2 = var_conv_kernel(ch_out, ch_out)

        # r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            h1 = tf.nn.max_pool3d(r1, k_pool, strides2, padding="SAME")
        else:
            w1 = var_conv_kernel(ch_out, ch_out)
            h1 = conv3_block(r1, w1, strides2, name='W1')
        return h1, r1


def upsample_Unet_block_1(input_1, input_2, ch_in, ch_out, k_conv0=None, name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    # strides2 = [1, 24, 16, 14, 1]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_2.shape.as_list()
    size_out[4] = size_out[4] * 2
    with tf.compat.v1.variable_scope(name):
        h0 = conv3_block(input_1, ch_in, ch_in, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_in, ch_out, name='WR1')

        r2 = deconv3_block(r1, ch_out, ch_out, size_out, strides2)
        # h0 = deconv3_block(input_, ch_out, ch_in, size_out, strides2)
        # input_2 = tf.pad(input_2, [[0, 0], [0, 4], [0, 0], [0, 2], [0, 0]])
        h1 = tf.concat([r2, input_2], axis=-1)
        return h1


def upsample_AFUnet_block_1(input_1, input_2, ch_in, ch_out, k_conv0=None, name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_2.shape.as_list()
    size_out[4] = size_out[4] * 2
    with tf.variable_scope(name):
        h0 = conv3_block(input_1, ch_in, ch_in, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_in, ch_out, name='WR1')

        r2 = deconv3_block(r1, ch_out, ch_out, size_out, strides2)
        # h0 = deconv3_block(input_, ch_out, ch_in, size_out, strides2)
        h1 = tf.concat([r2, input_2], axis=-1)
        return h1, h0, r1


def upsample_Unet_block(input_1, input_2, ch_in, ch_out, k_conv0=None, name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_2.shape.as_list()
    size_out[4] = size_out[4] * 2
    with tf.compat.v1.variable_scope(name):
        h0 = conv3_block(input_1, ch_in + ch_out, ch_out, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_out, ch_out, name='WR1')

        r2 = deconv3_block(r1, ch_out, ch_out, size_out, strides2)
        # h0 = deconv3_block(input_, ch_out, ch_in, size_out, strides2)
        h1 = tf.concat([r2, input_2], axis=-1)
        return h1

def downsample_MultiResUnet_block(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, use_respath=True,
                                  name='down_Unet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    with tf.variable_scope(name):
        shortcut = conv3_block(input_, ch_in, ch_out, k_conv=[1, 1, 1], name='cut')
        h1 = conv3_block(input_, ch_in, int(ch_out * 0.167) + 1, k_conv0, name='W1')
        h2 = conv3_block(h1, int(ch_out * 0.167) + 1, int(ch_out * 0.333), name='W2')
        h3 = conv3_block(h2, int(ch_out * 0.333), int(ch_out * 0.5), name='W3')
        out_concat = tf.contrib.layers.batch_norm(tf.concat([h1, h2, h3], axis=-1))
        out_add = tf.nn.relu(tf.contrib.layers.batch_norm((out_concat + shortcut)))
        if use_respath:

            w2 = var_conv_kernel(ch_out, ch_out)
            rp1_shortcut = conv3_block(out_add, ch_out, ch_out, k_conv=[1, 1, 1], name='rp1_sc')
            rp1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(out_add, w2, strides1, "SAME")) + rp1_shortcut)
            rp2_shortcut = conv3_block(rp1, ch_out, ch_out, k_conv=[1, 1, 1], name='rp2_sc')
            rp2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(rp1, w2, strides1, "SAME")) + rp2_shortcut)
            rp3_shortcut = conv3_block(rp2, ch_out, ch_out, k_conv=[1, 1, 1], name='rp3_sc')
            rp3 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(rp2, w2, strides1, "SAME")) + rp3_shortcut)
            rp4_shortcut = conv3_block(rp3, ch_out, ch_out, k_conv=[1, 1, 1], name='rp4_sc')
            rp4_one = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(rp3, w2, strides1, "SAME")) + rp4_shortcut)
        else:
            rp4_one = out_add
        if use_pooling:
            k_pool = [1, 2, 2, 2, 1]
            p1 = tf.nn.max_pool3d(out_add, k_pool, strides2, padding="SAME")
        else:
            p1 = out_add

        return p1, rp4_one


def upsample_MultiResUnet_block(input_1, input_2, ch_in, ch_out, k_conv0=None, use_upconv=True, use_concat=True,
                                name='up_MultiResUnet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_2.shape.as_list()
    size_out[4] = size_out[4] * 2
    with tf.variable_scope(name):
        shortcut = conv3_block(input_1, ch_in + ch_out, ch_out, k_conv=[1, 1, 1], name='cut')
        h1 = conv3_block(input_1, ch_in + ch_out, int(ch_out * 0.167) + 1, k_conv0, name='W1')
        h2 = conv3_block(h1, int(ch_out * 0.167) + 1, int(ch_out * 0.333), name='W2')
        h3 = conv3_block(h2, int(ch_out * 0.333), int(ch_out * 0.5), name='W3')
        out_concat = tf.contrib.layers.batch_norm(tf.concat([h1, h2, h3], axis=-1))
        out_add = tf.nn.relu(tf.contrib.layers.batch_norm((out_concat + shortcut)))
        if use_upconv:
            dc = deconv3_block(out_add, ch_out, ch_out, size_out, strides2)
        else:
            dc = out_add

        if use_concat:
            up = tf.concat([input_2, dc], axis=-1)
        else:
            up = out_add
        return up


def upsample_MultiResUnet_block_1(input_1, input_2, ch_in, ch_out, k_conv0=None, use_upconv=True, use_concat=True,
                                  name='up_MultiResUnet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides2 = [1, 2, 2, 2, 1]
    size_out = input_2.shape.as_list()
    size_out[4] = size_out[4] * 2
    with tf.variable_scope(name):
        shortcut = conv3_block(input_1, ch_in, ch_out, k_conv=[1, 1, 1], name='cut')
        h1 = conv3_block(input_1, ch_in, int(ch_out * 0.167) + 1, k_conv0, name='W1')
        h2 = conv3_block(h1, int(ch_out * 0.167) + 1, int(ch_out * 0.333), name='W2')
        h3 = conv3_block(h2, int(ch_out * 0.333), int(ch_out * 0.5), name='W3')
        out_concat = tf.contrib.layers.batch_norm(tf.concat([h1, h2, h3], axis=-1))
        out_add = tf.nn.relu(tf.contrib.layers.batch_norm((out_concat + shortcut)))
        if use_upconv:
            dc = deconv3_block(out_add, ch_out, ch_out, size_out, strides2)
        else:
            dc = out_add

        if use_concat:
            up = tf.concat([input_2, dc], axis=-1)
        else:
            up = out_add
        return up, out_add, h2, h1


def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    strides1 = [1, 1, 1, 1, 1]
    strides2 = [1, 2, 2, 2, 1]
    shape_x = x.shape.as_list()
    shape_g = g.shape.as_list()

    theta_x = conv3D(x, shape_x[4], inter_shape, k_conv=[2, 2, 2], strides=strides2, name='X1' + name)
    shape_theta_x = theta_x.shape.as_list()

    phi_g = conv3D(g, shape_g[4], inter_shape, k_conv=[1, 1, 1], name='X2' + name)
    upsample_g = deconv3_block(phi_g, shape_g[4], inter_shape, shape_theta_x, strides1, name='decov' + name)
    concat_xg = upsample_g + theta_x
    act_xg = tf.nn.relu(concat_xg)
    # tf.reset_default_graph()
    psi = conv3D(act_xg, shape_theta_x[4], 1, k_conv=[1, 1, 1], name='X3' + name)
    sigmoid_xg = tf.nn.sigmoid(psi)
    upsample_psi = tf.keras.layers.UpSampling3D(size=[2, 2, 2])(sigmoid_xg)
    upsample_psi = expend_as(upsample_psi, shape_x[4], name='epd' + name)
    y = tf.multiply(upsample_psi, x, name='mul' + name)
    shape_y = y.shape.as_list()
    result = conv3D(y, shape_y[4], shape_x[4], k_conv=[1, 1, 1], name='rconv' + name)
    result_bn = tf.compat.v1.layers.batch_normalization(result)
    return result_bn


def UnetGatingSignal(input, name='UnetGS'):
    shape = input.shape.as_list()
    x = conv3_block(input, shape[4], shape[4], k_conv=[1, 1, 1], name='gating' + name)
    return x


def expend_as(tensor, rep, name):
    my_repeat = tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=4),
                                       arguments={'repnum': rep},
                                       name='psi_up' + name)(tensor)
    return my_repeat


# No Local Unet code

def _output_block_layer(inputs, training, num_classes):
    inputs = BN_ReLU(inputs, training)

    inputs = tf.layers.dropout(inputs, rate=0.5, training=training)

    inputs = Conv3D(
        inputs=inputs,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        use_bias=True)

    return tf.identity(inputs, 'output')


def _encoding_block_layer(inputs, filters, block_fn,
                          blocks, strides, training, name):
    """Creates one layer of encoding blocks for the model.

        Args:
            inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
            filters: The number of filters for the first convolution of the layer.
            block_fn: The block to use within the model.
            blocks: The number of blocks contained in the layer.
            strides: The stride to use for the first convolution of the layer. If
                greater than 1, this layer will ultimately downsample the input.
            training: Either True or False, whether we are currently training the
                model. Needed for batch norm.
            name: A string name for the tensor output of the block layer.

        Returns:
            The output tensor of the block layer.
        """

    def projection_shortcut(inputs):
        return Conv3D(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=strides)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1)

    return tf.identity(inputs, name)


def _att_decoding_block_layer(inputs, skip_inputs, filters,
                              block_fn, blocks, strides, training, name):
    """Creates one layer of decoding blocks for the model.

        Args:
            inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
            skip_inputs: A tensor of size [batch, depth_in, height_in, width_in, filters].
            filters: The number of filters for the first convolution of the layer.
            block_fn: The block to use within the model.
            blocks: The number of blocks contained in the layer.
            strides: The stride to use for the first convolution of the layer. If
                greater than 1, this layer will ultimately downsample the input.
            training: Either True or False, whether we are currently training the
                model. Needed for batch norm.
            name: A string name for the tensor output of the block layer.

        Returns:
            The output tensor of the block layer.
        """

    def projection_shortcut(inputs):
        return Deconv3D(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides)

    inputs = _attention_block(inputs, filters, training, projection_shortcut, strides)

    inputs = inputs + skip_inputs

    for _ in range(0, blocks):
        inputs = block_fn(inputs, filters, training, None, 1)

    return tf.identity(inputs, name)


def _decoding_block_layer(inputs, skip_inputs, filters,
                          block_fn, blocks, strides, training, name):
    """Creates one layer of decoding blocks for the model.

        Args:
            inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
            skip_inputs: A tensor of size [batch, depth_in, height_in, width_in, filters].
            filters: The number of filters for the first convolution of the layer.
            block_fn: The block to use within the model.
            blocks: The number of blocks contained in the layer.
            strides: The stride to use for the first convolution of the layer. If
                greater than 1, this layer will ultimately downsample the input.
            training: Either True or False, whether we are currently training the
                model. Needed for batch norm.
            name: A string name for the tensor output of the block layer.

        Returns:
            The output tensor of the block layer.
        """

    inputs = Deconv3D(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides)

    inputs = inputs + skip_inputs

    for _ in range(0, blocks):
        inputs = block_fn(inputs, filters, training, None, 1)

    return tf.identity(inputs, name)


################################################################################
# Basic blocks building the network
################################################################################
def _residual_block(inputs, filters, training,
                    projection_shortcut, strides):
    """Standard building block for residual networks with BN before convolutions.

        Args:
            inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference
                mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
                downsample the input.

        Returns:
            The output tensor of the block.
        """

    shortcut = inputs
    inputs = BN_ReLU(inputs, training)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = Conv3D(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides)

    inputs = BN_ReLU(inputs, training)

    inputs = Conv3D(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1)
    return inputs + shortcut


def _attention_block(inputs, filters, training,
                     projection_shortcut, strides):
    """Attentional building block for residual networks with BN before convolutions.

        Args:
            inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference
                mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
                downsample the input.

        Returns:
            The output tensor of the block.
        """

    shortcut = inputs
    inputs = BN_ReLU(inputs, training)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    if strides != 1:
        layer_type = 'UP'
    else:
        layer_type = 'SAME'

    inputs = multihead_attention_3d(
        inputs, filters, filters, filters, 1, training, layer_type)

    return inputs + shortcut


def Pool3d(inputs, kernel_size, strides):
    """Performs 3D max pooling."""

    return tf.layers.max_pooling3d(
        inputs=inputs,
        pool_size=kernel_size,
        strides=strides,
        padding='same')


def Deconv3D(inputs, filters, kernel_size, strides, use_bias=False):
    """Performs 3D deconvolution without bias and activation function."""

    return tf.layers.conv3d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer())


def Conv3D(inputs, filters, kernel_size, strides, use_bias=False):
    """Performs 3D convolution without bias and activation function."""

    return tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer())


def Dilated_Conv3D(inputs, filters, kernel_size, dilation_rate, use_bias=False):
    """Performs 3D dilated convolution without bias and activation function."""

    return tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        dilation_rate=dilation_rate,
        padding='same',
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer())


def BN_ReLU(inputs, training):
    """Performs a batch normalization followed by a ReLU6."""

    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        momentum=0.997,
        epsilon=1e-5,
        center=True,
        scale=True,
        training=training,
        fused=True)

    return tf.nn.relu6(inputs)


def multihead_attention_3d(inputs, total_key_filters, total_value_filters,
                           output_filters, num_heads, training, layer_type='SAME',
                           name=None):
    """3d Multihead scaled-dot-product attention with input/output transformations.

    Args:
        inputs: a Tensor with shape [batch, d, h, w, channels]
        total_key_filters: an integer. Note that queries have the same number
            of channels as keys
        total_value_filters: an integer
        output_depth: an integer
        num_heads: an integer dividing total_key_filters and total_value_filters
        layer_type: a string, type of this layer -- SAME, DOWN, UP
        name: an optional string

    Returns:
        A Tensor of shape [batch, _d, _h, _w, output_filters]

    Raises:
        ValueError: if the total_key_filters or total_value_filters are not divisible
            by the number of attention heads.
    """

    if total_key_filters % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_key_filters, num_heads))
    if total_value_filters % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_filters, num_heads))
    if layer_type not in ['SAME', 'DOWN', 'UP']:
        raise ValueError("Layer type (%s) must be one of SAME, "
                         "DOWN, UP." % (layer_type))

    with tf.variable_scope(
            name,
            default_name="multihead_attention_3d",
            values=[inputs]):

        # produce q, k, v
        q, k, v = compute_qkv_3d(inputs, total_key_filters,
                                 total_value_filters, layer_type)

        # after splitting, shape is [batch, heads, d, h, w, channels / heads]
        q = split_heads_3d(q, num_heads)
        k = split_heads_3d(k, num_heads)
        v = split_heads_3d(v, num_heads)

        # normalize
        key_filters_per_head = total_key_filters // num_heads
        q *= key_filters_per_head ** -0.5

        # attention
        x = global_attention_3d(q, k, v, training)

        x = combine_heads_3d(x)
        x = Conv3D(x, output_filters, 1, 1, use_bias=True)

        return x


def compute_qkv_3d(inputs, total_key_filters, total_value_filters, layer_type):
    """Computes query, key and value.

    Args:
        inputs: a Tensor with shape [batch, d, h, w, channels]
        total_key_filters: an integer
        total_value_filters: and integer
        layer_type: String, type of this layer -- SAME, DOWN, UP

    Returns:
        q: [batch, _d, _h, _w, total_key_filters] tensor
        k: [batch, h, w, total_key_filters] tensor
        v: [batch, h, w, total_value_filters] tensor
    """

    # linear transformation for q
    if layer_type == 'SAME':
        q = Conv3D(inputs, total_key_filters, 1, 1, use_bias=True)
    elif layer_type == 'DOWN':
        q = Conv3D(inputs, total_key_filters, 3, 2, use_bias=True)
    elif layer_type == 'UP':
        q = Deconv3D(inputs, total_key_filters, 3, 2, use_bias=True)

    # linear transformation for k
    k = Conv3D(inputs, total_key_filters, 1, 1, use_bias=True)

    # linear transformation for k
    v = Conv3D(inputs, total_value_filters, 1, 1, use_bias=True)

    return q, k, v


def split_heads_3d(x, num_heads):
    """Split channels (last dimension) into multiple heads (becomes dimension 1).

    Args:
        x: a Tensor with shape [batch, d, h, w, channels]
        num_heads: an integer

    Returns:
        a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]
    """

    return tf.transpose(split_last_dimension(x, num_heads), [0, 4, 1, 2, 3, 5])


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.

    Args:
        x: a Tensor with shape [..., m]
        n: an integer.

    Returns:
        a Tensor with shape [..., n, m/n]
    """

    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]

    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)

    return ret


def global_attention_3d(q, k, v, training, name=None):
    """global self-attention.
    Args:
        q: a Tensor with shape [batch, heads, _d, _h, _w, channels_k]
        k: a Tensor with shape [batch, heads, d, h, w, channels_k]
        v: a Tensor with shape [batch, heads, d, h, w, channels_v]
        name: an optional string
    Returns:
        a Tensor of shape [batch, heads, _d, _h, _w, channels_v]
    """
    with tf.variable_scope(
            name,
            default_name="global_attention_3d",
            values=[q, k, v]):
        new_shape = tf.concat([tf.shape(q)[0:-1], [v.shape[-1].value]], 0)

        # flatten q,k,v
        q_new = flatten_3d(q)
        k_new = flatten_3d(k)
        v_new = flatten_3d(v)

        # attention
        output = dot_product_attention(q_new, k_new, v_new, bias=None,
                                       training=training, dropout_rate=0.5, name="global_3d")

        # putting the representations back in the right place
        output = scatter_3d(output, new_shape)

        return output


def reshape_range(tensor, i, j, shape):
    """Reshapes a tensor between dimensions i and j."""

    target_shape = tf.concat(
        [tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
        axis=0)

    return tf.reshape(tensor, target_shape)


def flatten_3d(x):
    """flatten x."""

    x_shape = tf.shape(x)
    # [batch, heads, length, channels], length = d*h*w
    x = reshape_range(x, 2, 5, [tf.reduce_prod(x_shape[2:5])])

    return x


def scatter_3d(x, shape):
    """scatter x."""

    x = tf.reshape(x, shape)

    return x


def dot_product_attention(q, k, v, bias, training, dropout_rate=0.0, name=None):
    """Dot-product attention.

    Args:
        q: a Tensor with shape [batch, heads, length_q, channels_k]
        k: a Tensor with shape [batch, heads, length_kv, channels_k]
        v: a Tensor with shape [batch, heads, length_kv, channels_v]
        bias: bias Tensor
        dropout_rate: a floating point number
        name: an optional string

    Returns:
        A Tensor with shape [batch, heads, length_q, channels_v]
    """

    with tf.variable_scope(
            name,
            default_name="dot_product_attention",
            values=[q, k, v]):
        # [batch, num_heads, length_q, length_kv]
        logits = tf.matmul(q, k, transpose_b=True)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        # dropping out the attention links for each of the heads
        weights = tf.layers.dropout(weights, dropout_rate, training)

        return tf.matmul(weights, v)


def combine_heads_3d(x):
    """Inverse of split_heads_3d.

    Args:
        x: a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]

    Returns:
        a Tensor with shape [batch, d, h, w, channels]
    """

    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 4, 1, 5]))


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.

    Args:
        x: a Tensor with shape [..., a, b]

    Returns:
        a Tensor with shape [..., a*b]
    """

    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]

    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)

    return ret


# FCN-code
nc = [16, 32, 64, 128, 256]


def vgg_net(input, channels, name='vgg_net_pre'):
    k_conv0 = [3, 3, 3]
    k_pool = [1, 2, 2, 2, 1]
    strides2 = [1, 2, 2, 2, 1]

    conv11 = conv3_block(input, 2, channels[0], k_conv=[7, 7, 7], name='conv11' + name)
    conv12 = conv3_block(conv11, channels[0], channels[0], k_conv0, name='conv12' + name)
    pool11 = tf.nn.max_pool3d(conv12, k_pool, strides2, padding="SAME")

    conv21 = conv3_block(pool11, channels[0], channels[1], k_conv0, name='conv21' + name)
    conv22 = conv3_block(conv21, channels[1], channels[1], k_conv0, name='conv22' + name)
    pool21 = tf.nn.max_pool3d(conv22, k_pool, strides2, padding="SAME")

    conv31 = conv3_block(pool21, channels[1], channels[2], k_conv0, name='conv31' + name)
    conv32 = conv3_block(conv31, channels[2], channels[2], k_conv0, name='conv32' + name)
    conv33 = conv3_block(conv32, channels[2], channels[2], k_conv0, name='conv33' + name)
    pool31 = tf.nn.max_pool3d(conv33, k_pool, strides2, padding="SAME")

    conv41 = conv3_block(pool31, channels[2], channels[3], k_conv0, name='conv41' + name)
    conv42 = conv3_block(conv41, channels[3], channels[3], k_conv0, name='conv42' + name)
    conv43 = conv3_block(conv42, channels[3], channels[3], k_conv0, name='conv43' + name)
    pool41 = tf.nn.max_pool3d(conv43, k_pool, strides2, padding="SAME")

    conv51 = conv3_block(pool41, channels[3], channels[3], k_conv0, name='conv51' + name)
    conv52 = conv3_block(conv51, channels[3], channels[3], k_conv0, name='conv52' + name)
    conv53 = conv3_block(conv52, channels[3], channels[3], k_conv0, name='conv53' + name)
    pool51 = tf.nn.max_pool3d(conv53, k_pool, strides2, padding="SAME")

    return pool31, pool41, pool51


def fcndeconv3_block(input_, ch_in, ch_out, shape_out, strides, k_conv, name='deconv3_block'):
    with tf.variable_scope(name):
        w = var_conv_kernel(ch_in, ch_out, k_conv)
        return tf.nn.relu(
            tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(input_, w, shape_out, strides, "SAME")))


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


# Vnet-code
def xavier_initializer_convolution(shape, dist='uniform', lambda_initializer=True):
    """
    Xavier initializer for N-D convolution patches. input_activations = patch_volume * in_channels;
    output_activations = patch_volume * out_channels; Uniform: lim = sqrt(3/(input_activations + output_activations))
    Normal: stddev =  sqrt(6/(input_activations + output_activations))
    :param shape: The shape of the convolution patch i.e. spatial_shape + [input_channels, output_channels]. The order of
    input_channels and output_channels is irrelevant, hence this can be used to initialize deconvolution parameters.
    :param dist: A string either 'uniform' or 'normal' determining the type of distribution
    :param lambda_initializer: Whether to return the initial actual values of the parameters (True) or placeholders that
    are initialized when the session is initiated
    :return: A numpy araray with the initial values for the parameters in the patch
    """
    s = len(shape) - 2
    num_activations = np.prod(shape[:s]) * np.sum(shape[s:])  # input_activations + output_activations
    if dist == 'uniform':
        lim = np.sqrt(6. / num_activations)
        if lambda_initializer:
            return np.random.uniform(-lim, lim, shape).astype(np.float32)
        else:
            return tf.random_uniform(shape, minval=-lim, maxval=lim)
    if dist == 'normal':
        stddev = np.sqrt(3. / num_activations)
        if lambda_initializer:
            return np.random.normal(0, stddev, shape).astype(np.float32)
        else:
            tf.truncated_normal(shape, mean=0, stddev=stddev)
    raise ValueError('Distribution must be either "uniform" or "normal".')


def constant_initializer(value, shape, lambda_initializer=True):
    if lambda_initializer:
        return np.full(shape, value).astype(np.float32)
    else:
        return tf.constant(value, tf.float32, shape)


def get_spatial_rank(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the spatial rank of the tensor i.e. the number of spatial dimensions between batch_size and num_channels
    """
    return len(x.get_shape()) - 2


def get_num_channels(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the number of channels of x
    """
    return int(x.get_shape()[-1])


def get_spatial_size(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: The spatial shape of x, excluding batch_size and num_channels.
    """
    return x.get_shape()[1:-1]


# parametric leaky relu
def prelu(x):
    alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def convolution(x, filter, padding='SAME', strides=None, dilation_rate=None):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-1]))

    return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b


def deconvolution(x, filter, output_shape, strides, padding='SAME'):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-2]))

    spatial_rank = get_spatial_rank(x)
    if spatial_rank == 2:
        return tf.nn.conv2d_transpose(x, filter, output_shape, strides, padding) + b
    if spatial_rank == 3:
        return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b
    raise ValueError('Only 2D and 3D images supported.')


# More complex blocks

# down convolution
def down_convolution(x, factor, kernel_size):
    num_channels = get_num_channels(x)
    spatial_rank = get_spatial_rank(x)
    strides = spatial_rank * [factor]
    filter = kernel_size + [num_channels, num_channels * factor]
    x = convolution(x, filter, strides=strides)
    return x


# up convolution
def up_convolution(x, output_shape, factor, kernel_size):
    num_channels = get_num_channels(x)
    spatial_rank = get_spatial_rank(x)
    strides = [1] + spatial_rank * [factor] + [1]
    filter = kernel_size + [num_channels // factor, num_channels]
    x = deconvolution(x, filter, output_shape, strides=strides)
    return x


def convolution_block(layer_input, num_convolutions, keep_prob, activation_fn, is_training):
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        with tf.variable_scope('conv_' + str(i + 1)):
            x = convolution(x, [5, 5, 5, n_channels, n_channels])
            layer_input = tf.layers.batch_normalization(layer_input, momentum=0.99, epsilon=0.001, center=True,
                                                        scale=True, training=is_training)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
    return x


def convolution_block_2(layer_input, fine_grained_features, num_convolutions, keep_prob, activation_fn, is_training):
    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    n_channels = get_num_channels(layer_input)
    if num_convolutions == 1:
        with tf.variable_scope('conv_' + str(1)):
            x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              training=is_training)
            layer_input = tf.layers.batch_normalization(layer_input, momentum=0.99, epsilon=0.001, center=True,
                                                        scale=True, training=is_training)
            x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
        return x

    with tf.variable_scope('conv_' + str(1)):
        x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                          training=is_training)
        x = activation_fn(x)
        x = tf.nn.dropout(x, keep_prob)

    for i in range(1, num_convolutions):
        with tf.variable_scope('conv_' + str(i + 1)):
            x = convolution(x, [5, 5, 5, n_channels, n_channels])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              training=is_training)
            layer_input = tf.layers.batch_normalization(layer_input, momentum=0.99, epsilon=0.001, center=True,
                                                        scale=True, training=is_training)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)

    return x
