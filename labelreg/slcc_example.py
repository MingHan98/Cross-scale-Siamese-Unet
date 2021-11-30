import tensorflow as tf
import numpy as np
import keras.layers as KL
import nibabel as nib
import os
import losses

data_path = r"/data/smh/data/jdm_data/cross1/test/TOF_images"


image_fixed = os.path.join(data_path, 'case000001.nii.gz')
image_fixed_load = nib.load(image_fixed)
image_fixed_data = image_fixed_load.get_data()
image_fixed_data = np.expand_dims(np.expand_dims(image_fixed_data, 0), -1)

image_fixed_label = os.path.join(data_path, 'case000001_label.nii.gz')
image_fixed_label_load = nib.load(image_fixed_label)
image_fixed_label_data = image_fixed_label_load.get_data().astype(np.float32)
image_fixed_label_data = tf.convert_to_tensor(np.expand_dims(np.expand_dims(image_fixed_label_data, 0), -1))

image_move = os.path.join(data_path, 'case000001_move.nii.gz')
image_move_load = nib.load(image_move)
image_move_data = image_move_load.get_data()
image_move_data = tf.convert_to_tensor(np.expand_dims(np.expand_dims(image_move_data, 0), -1))

image_move_label = os.path.join(data_path, 'case000001move_label.nii.gz')
image_move_label_load = nib.load(image_move_label)
image_move_label_data = image_move_label_load.get_data().astype(np.float32)
image_move_label_data = tf.convert_to_tensor(np.expand_dims(np.expand_dims(image_move_label_data, 0), -1))

fixed_image = tf.cast(tf.convert_to_tensor(image_fixed_data), dtype=tf.float32)
move_image = tf.cast(tf.convert_to_tensor(image_move_data), dtype=tf.float32)
# a = sparse_conv_cc3D()
slcc_value = losses.sparse_conv_cc3D(I, J, I_mask, atlas_mask)
with tf.Session() as sess:
    print (sess.run(slcc_value))
# image_warped = os.path.join(data_path, 'warped_image0.nii.gz')
# image_warped_load = nib.load(image_warped)
# image_warped_data = image_warped_load.get_data()
# image_warped_data = np.expand_dims(np.expand_dims(image_warped_data, 0), -1)
#
# image_warped_label = os.path.join(data_path, 'warped_label0.nii.gz')
# image_warped_label_load = nib.load(image_warped_label)
# image_warped_label_data = image_warped_label_load.get_data()
# image_warped_label_data = np.expand_dims(np.expand_dims(image_warped_label_data, 0), -1)
#
# image_warped1 = os.path.join(data_path, 'warped_image3.nii.gz')
# image_warped1_load = nib.load(image_warped1)
# image_warped1_data = image_warped1_load.get_data()
# image_warped1_data = np.expand_dims(np.expand_dims(image_warped1_data, 0), -1)
#
# image_warped1_label = os.path.join(data_path, 'warped_label3.nii.gz')
# image_warped1_label_load = nib.load(image_warped1_label)
# image_warped1_label_data = image_warped1_label_load.get_data()
# image_warped1_label_data = np.expand_dims(np.expand_dims(image_warped1_label_data, 0), -1)














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
    out_data = conv_data/(conv_mask+1e-2)
    mask_norm = lambda x: tf.cast(x > 0, tf.float32)
    # out_mask = tf.keras.layers.Lambda(mask_norm, name='%s_norm_wt' % core_name)(conv_mask)
    out_mask = mask_norm(conv_mask)
    return (out_data, out_mask, conv_data, conv_mask)


def sparse_conv_cc3D(I, J, I_mask, atlas_mask, conv_size=3, sum_filter=1, padding='same', slcc_weight=1):
    '''
        Sparse Normalized Local Cross Correlation (SLCC) for 3D images
    '''
    # pass in mask to class: e.g. Mask(model.get_layer("mask").output).sparse_conv_cc3D(atlas_mask),
    mask = I_mask
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

    combined_mask = mask * atlas_mask
    u_I, out_mask_I, not_used, conv_mask_I = conv_block(I, mask, im_conv, mask_conv, 'u_I')
    u_J, out_mask_J, not_used, conv_mask_J = conv_block(J, atlas_mask, im_conv, mask_conv, 'u_J')
    not_used, not_used_mask, I_sum, conv_mask = conv_block(I, combined_mask, im_conv, mask_conv, 'I_sum')
    not_used, not_used_mask, J_sum, conv_mask = conv_block(J, combined_mask, im_conv, mask_conv, 'J_sum')
    not_used, not_used_mask, I2_sum, conv_mask = conv_block(I2, combined_mask, im_conv, mask_conv, 'I2_sum')
    not_used, not_used_mask, J2_sum, conv_mask = conv_block(J2, combined_mask, im_conv, mask_conv, 'J2_sum')
    not_used, not_used_mask, IJ_sum, conv_mask = conv_block(IJ, combined_mask, im_conv, mask_conv, 'IJ_sum')
    # conv_mask
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * conv_mask
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * conv_mask
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * conv_mask

    cc = (cross * cross / (I_var * J_var + 1e-2))*slcc_weight
    # return -1.0 * tf.reduce_mean(cc)

    return tf.reduce_mean(cc)


