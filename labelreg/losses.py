import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as KL


def build_loss(similarity_type, similarity_scales, regulariser_type, regulariser_weight,
               label_moving, label_fixed, network_type, ddf):
    # label_similarity = compute_binary_dice(label_moving, label_fixed)
    label_similarity = multi_scale_loss(label_fixed, label_moving, similarity_type.lower(), similarity_scales)
    if network_type.lower() == 'global':
        ddf_regularisation = tf.constant(0.0)
    else:
        ddf_regularisation = tf.reduce_mean(local_displacement_energy(ddf, regulariser_type, regulariser_weight))

    return tf.reduce_mean(label_similarity), ddf_regularisation


def build_loss_1(similarity_type, similarity_scales, regulariser_type, regulariser_weight, label_moving, label_fixed,
                 network_type, ddf, image_warped, image_fixed):
    # label_similarity = compute_binary_dice(label_moving, label_fixed)
    label_similarity = multi_scale_loss(label_fixed, label_moving, similarity_type.lower(), similarity_scales)
    if network_type.lower() == 'global':
        ddf_regularisation = tf.constant(0.0)
    else:
        ddf_regularisation = tf.reduce_mean(local_displacement_energy(ddf, regulariser_type, regulariser_weight))
    slcc_cross = sparse_conv_cc3D(image_fixed, image_warped, label_fixed, label_moving)

    return tf.reduce_mean(label_similarity), ddf_regularisation, slcc_cross


def build_loss_2(similarity_type, similarity_scales, regulariser_type, regulariser_weight, label_moving, label_fixed,
                 network_type, ddf):
    # label_similarity = compute_binary_dice(label_moving, label_fixed)
    label_similarity = multi_scale_loss(label_fixed, label_moving, similarity_type.lower(), similarity_scales)
    label_distance = compute_centroid_distance(label_moving, label_fixed)
    if network_type.lower() == 'global':
        ddf_regularisation = tf.constant(0.0)
    else:
        ddf_regularisation = tf.reduce_mean(local_displacement_energy(ddf, regulariser_type, regulariser_weight))

    return tf.reduce_mean(label_similarity), ddf_regularisation, tf.sigmoid(tf.reduce_mean(label_distance))

def weighted_binary_cross_entropy(ts, ps, pw=1, eps=1e-6):
    ps = tf.clip_by_value(ps, eps, 1 - eps)
    return -tf.reduce_sum(
        tf.concat([ts * pw, 1 - ts], axis=4) * tf.log(tf.concat([ps, 1 - ps], axis=4)),
        axis=4, keep_dims=True)


def dice_simple(ts, ps, eps_vol=1e-6):

    # ps = tf.pad(ps, [[0, 0], [0, 16], [0, 0], [0, 8], [0, 0]])

    numerator = tf.reduce_sum(ts * ps, axis=[1, 2, 3, 4]) * 2
    denominator = tf.reduce_sum(ts, axis=[1, 2, 3, 4]) + tf.reduce_sum(ps, axis=[1, 2, 3, 4]) + eps_vol
    return numerator / denominator


def dice_generalised(ts, ps, weights):
    ts2 = tf.concat([ts, 1 - ts], axis=4)
    ps2 = tf.concat([ps, 1 - ps], axis=4)
    numerator = 2 * tf.reduce_sum(tf.reduce_sum(ts2 * ps2, axis=[1, 2, 3]) * weights, axis=1)
    denominator = tf.reduce_sum((tf.reduce_sum(ts2, axis=[1, 2, 3]) +
                                 tf.reduce_sum(ps2, axis=[1, 2, 3])) * weights, axis=1)
    return numerator / denominator



def jaccard_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts * ps, axis=[1, 2, 3, 4])
    denominator = tf.reduce_sum(tf.square(ts), axis=[1, 2, 3, 4]) + \
                  tf.reduce_sum(tf.square(ps), axis=[1, 2, 3, 4]) - numerator + eps_vol
    return numerator / denominator


def gauss_kernel1d(sigma):
    if sigma == 0:
        return 0
    else:
        tail = int(sigma * 3)
        k = tf.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])

        return k / tf.reduce_sum(k)


def cauchy_kernel1d(sigma):  # this is an approximation
    if sigma == 0:
        return 0
    else:
        tail = int(sigma * 5)
        # k = tf.reciprocal(([((x/sigma)**2+1)*sigma*3.141592653589793 for x in range(-tail, tail+1)]))
        k = tf.reciprocal([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def separable_filter3d(vol, kernel):
    if kernel == 0:
        return vol
    else:
        strides = [1, 1, 1, 1, 1]

        return (tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
            vol,

            tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, -1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, 1, -1, 1, 1]), strides, "SAME"))


def single_scale_loss(label_fixed, label_moving, loss_type):
    if loss_type == 'cross-entropy':
        label_loss_batch = tf.reduce_mean(weighted_binary_cross_entropy(label_fixed, label_moving), axis=[1, 2, 3, 4])
    elif loss_type == 'mean-squared':
        label_loss_batch = tf.reduce_mean(tf.squared_difference(label_fixed, label_moving), axis=[1, 2, 3, 4])
    elif loss_type == 'dice':

        label_loss_batch = 1 - dice_simple(label_fixed, label_moving)
    elif loss_type == 'jaccard':
        label_loss_batch = 1 - jaccard_simple(label_fixed, label_moving)
    else:
        raise Exception('Not recognised label correspondence loss!')
    return label_loss_batch


def multi_scale_loss(label_fixed, label_moving, loss_type, loss_scales):
    label_loss_all = tf.stack(
        [single_scale_loss(
            separable_filter3d(label_fixed, gauss_kernel1d(s)),
            separable_filter3d(label_moving, gauss_kernel1d(s)), loss_type)
            for s in loss_scales],
        axis=1)
    return tf.reduce_mean(label_loss_all, axis=1)


def dice_loss(label_fixed, label_moving, loss_type):
    label_loss_all = single_scale_loss(label_fixed, label_moving, loss_type) * 0.5
    return tf.reduce_mean(label_loss_all, axis=0)


def local_displacement_energy(ddf, energy_type, energy_weight):

    def gradient_dx(fv):
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(fv):
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(fv):
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2


    def gradient_txyz(Txyz, fn):

        return tf.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], axis=4)

    def compute_gradient_norm(displacement, flag_l1=False):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        if flag_l1:
            norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
        else:
            norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
        return tf.reduce_mean(norms, [1, 2, 3, 4])

    def compute_bending_energy(displacement):

        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        dTdxx = gradient_txyz(dTdx, gradient_dx)
        dTdyy = gradient_txyz(dTdy, gradient_dy)
        dTdzz = gradient_txyz(dTdz, gradient_dz)
        dTdxy = gradient_txyz(dTdx, gradient_dy)
        dTdyz = gradient_txyz(dTdy, gradient_dz)
        dTdxz = gradient_txyz(dTdx, gradient_dz)

        return tf.reduce_mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2,
                              [1, 2, 3, 4])


    if energy_weight:
        if energy_type == 'bending':
            energy = compute_bending_energy(ddf)
        elif energy_type == 'gradient-l2':
            energy = compute_gradient_norm(ddf)
        elif energy_type == 'gradient-l1':
            energy = compute_gradient_norm(ddf, flag_l1=True)
        elif energy_type == 'jac':
            energy = Get_Jac(ddf)
        else:
            raise Exception('Not recognised local regulariser!')
    else:
        energy = tf.constant(0.0)

    return energy * energy_weight


def Get_Jac(displacement):
    '''
    the expected input: displacement of shape(batch, H, W, D, channel),
    obtained in TensorFlow.
    '''
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    D = D1 - D2 + D3
    return D



def compute_binary_dice(input1, input2):
    mask1 = input1 >= 0.5
    mask2 = input2 >= 0.5
    vol1 = tf.reduce_sum(tf.to_float(mask1), axis=[1, 2, 3, 4])
    vol2 = tf.reduce_sum(tf.to_float(mask2), axis=[1, 2, 3, 4])
    dice = tf.reduce_sum(tf.to_float(mask1 & mask2), axis=[1, 2, 3, 4]) * 2 / (vol1 + vol2)
    return dice


# class SparseVM(object):
#     '''
#     SparseVM Sparse Normalized Local Cross Correlation (SLCC)
#     '''
#
#     def __init__(self, mask):
#         self.mask = mask


# def conv_block(data, mask, conv_layer, mask_conv_layer, core_name):
#     wt_data = data*mask
#     conv_data = conv3D(wt_data, )
#
#     convL = getattr(KL, 'Conv%dD' % ndims)
#     im_conv = convL(sum_filter, conv_size, padding=padding, strides=strides, kernel_initializer=tf.keras.initializers.Ones())
#     im_conv.trainable = False

def conv_block(data, mask, conv_layer, mask_conv_layer, core_name):
    wt_data = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name='%s_pre_wmult' % core_name)([data, mask])
    # convolve data
    conv_data = conv_layer(wt_data)

    # convolve mask
    conv_mask = mask_conv_layer(mask)
    zero_mask = tf.keras.layers.Lambda(lambda x: x * 0 + 1)(mask)
    conv_mask_allones = mask_conv_layer(zero_mask)  # all_ones mask to get the edge counts right.
    mask_conv_layer.trainable = False
    o = np.ones(mask_conv_layer.get_weights()[0].shape)
    mask_conv_layer.set_weights([o])
    # re-weight data (this is what makes the conv makes sense)
    # data_norm = lambda x: x[0] / (x[1] + 1e-2)
    # data_norm = lambda x: x[0] / K.maximum(x[1]/x[2], 1)
    # out_data = tf.keras.layers.Lambda(data_norm, name='%s_norm_im' % core_name)([conv_data, conv_mask])
    out_data = conv_data / (conv_mask + 1e-2)
    mask_norm = lambda x: tf.cast(x > 0, tf.float32)
    # out_mask = tf.keras.layers.Lambda(mask_norm, name='%s_norm_wt' % core_name)(conv_mask)
    out_mask = mask_norm(conv_mask)

    return (out_data, out_mask, conv_data, conv_mask)


def conv_block_1(data, mask, conv_layer, mask_conv_layer, core_name):
    wt_data = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name='%s_pre_wmult' % core_name)([data, mask])
    # convolve data
    conv_data = conv_layer(wt_data)

    # convolve mask
    conv_mask = mask_conv_layer(mask)
    zero_mask = tf.keras.layers.Lambda(lambda x: x * 0 + 1)(mask)
    conv_mask_allones = mask_conv_layer(zero_mask)  # all_ones mask to get the edge counts right.
    mask_conv_layer.trainable = False
    o = np.ones(mask_conv_layer.get_weights()[0].shape)
    mask_conv_layer.set_weights([o])
    # re-weight data (this is what makes the conv makes sense)
    # data_norm = lambda x: x[0] / (x[1] + 1e-2)
    # data_norm = lambda x: x[0] / K.maximum(x[1]/x[2], 1)
    # out_data = tf.keras.layers.Lambda(data_norm, name='%s_norm_im' % core_name)([conv_data, conv_mask])
    out_data = conv_data / (conv_mask + 1e-2)
    mask_norm = lambda x: tf.cast(x > 0, tf.float32)
    # out_mask = tf.keras.layers.Lambda(mask_norm, name='%s_norm_wt' % core_name)(conv_mask)
    out_mask = mask_norm(conv_mask)

    return (out_data, out_mask, conv_data, conv_mask)


def sparse_conv_cc3D(I, J, I_mask, atlas_mask, conv_size=3, sum_filter=1, padding='same', slcc_weight=1):
    '''
        Sparse Normalized Local Cross Correlation (SLCC) for 3D images
    '''

    # pass in mask to class: e.g. Mask(model.get_layer("mask").output).sparse_conv_cc3D(atlas_mask),

    # need the next two lines to specify channel for source image (otherwise won't compile)
    I = I[:, :, :, :, 0]
    I = tf.expand_dims(I, -1)
    I2 = I * I
    J2 = J * J
    IJ = I * J
    input_shape = I.shape
    # want the size without the channel and batch dimensions
    ndims = len(input_shape) - 2
    strides = [1] * ndims

    convL = getattr(KL, 'Conv%dD' % ndims)
    im_conv = convL(sum_filter, conv_size, padding=padding, strides=strides,
                    kernel_initializer=tf.keras.initializers.Ones())
    im_conv.trainable = False
    mask_conv = convL(1, conv_size, padding=padding, use_bias=False, strides=strides,
                      kernel_initializer=tf.keras.initializers.Ones())
    mask_conv.trainable = False

    combined_mask = I_mask * atlas_mask
    u_I, out_mask_I, not_used, conv_mask_I = conv_block(I, I_mask, im_conv, mask_conv, 'u_I')
    u_J, out_mask_J, not_used, conv_mask_J = conv_block(J, atlas_mask, im_conv, mask_conv, 'u_J')
    not_used, not_used_mask, I_sum, conv_mask = conv_block(I, combined_mask, im_conv, mask_conv, 'I_sum')
    not_used, not_used_mask, J_sum, conv_mask = conv_block(J, combined_mask, im_conv, mask_conv, 'J_sum')
    not_used, not_used_mask, I2_sum, conv_mask = conv_block(I2, combined_mask, im_conv, mask_conv, 'I2_sum')
    not_used, not_used_mask, J2_sum, conv_mask = conv_block(J2, combined_mask, im_conv, mask_conv, 'J2_sum')
    not_used, not_used_mask, IJ_sum, conv_mask = conv_block(IJ, combined_mask, im_conv, mask_conv, 'IJ_sum')
    # conv_mask
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J*conv_mask
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I*conv_mask
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J*conv_mask

    cc = (cross * cross / (I_var * J_var + 1e-2)) * slcc_weight
    return tf.reduce_mean(cc)


def get_reference_grid(grid_size):
    return tf.cast(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3), dtype=tf.float32)

def compute_centroid_distance(input1, input2, grid=None):
    if grid is None:
        grid = get_reference_grid(input1.get_shape()[1:4])

    def compute_centroid(mask, grid0):

        return tf.stack([tf.reduce_mean(tf.boolean_mask(grid0, mask[i, ..., 0] >= 0.5), axis=0)
                         for i in range(mask.shape[0].value)], axis=0)

    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1 - c2), axis=1))



