import tensorflow as tf
# from tensorflow.python.ops import array_ops, math_ops
import numpy as np

def warp_image_affine(vol, theta):  # vol=image[3,64,64,60,1]  theat=affine[3,1,12]
    return resample_linear(vol, warp_grid(get_reference_grid(vol.get_shape()[1:4]), theta))


def warp_grid(grid, theta):
    # grid=grid_reference
    num_batch = int(theta.get_shape()[0])

    theta = tf.reshape(theta, (-1, 3, 4))
    size = grid.get_shape().as_list()

    grid = tf.concat([tf.transpose(tf.reshape(grid, [-1, 3])), tf.ones([1, size[0] * size[1] * size[2]])],
                     axis=0)
    grid = tf.reshape(tf.tile(tf.reshape(grid, [-1]), [num_batch]), [num_batch, 4, -1])

    grid_warped = tf.matmul(theta, grid)
    return tf.reshape(tf.transpose(grid_warped, [0, 2, 1]), [num_batch, size[0], size[1], size[2], 3])


def af_warp_grid(grid, theta):
    # grid=grid_reference
    num_batch = int(theta.get_shape()[0])

    theta = tf.reshape(theta, (-1, 3, 4))
    # a = tf.constant(1, dtype=tf.float32)
    # theta[0, 0, 0] = a
    # theta[0, 1, 1] = a
    # theta[0, 2, 2] = a
    # theta[1, 0, 0] = a
    # theta[1, 1, 1] = a
    # theta[1, 2, 2] = a
    # tf.assign(theta[0, 0, 0], a)
    # tf.assign(theta[0, 1, 1], a)
    # tf.assign(theta[0, 2, 2], a)
    # tf.assign(theta[1, 0, 0], a)
    # tf.assign(theta[1, 1, 1], a)
    # tf.assign(theta[1, 2, 2], a)

    size = grid.get_shape().as_list()

    grid = tf.concat([tf.transpose(tf.reshape(grid, [-1, 3])), tf.ones([1, size[0] * size[1] * size[2]])],
                     axis=0)
    grid = tf.reshape(tf.tile(tf.reshape(grid, [-1]), [num_batch]), [num_batch, 4, -1])


    grid_warped = tf.matmul(theta, grid)
    return tf.reshape(tf.transpose(grid_warped, [0, 2, 1]), [num_batch, size[0], size[1], size[2], 3])


def resample_linear(inputs, sample_coords):

    input_size = inputs.get_shape().as_list()[1:-1]
    spatial_rank = inputs.get_shape().ndims - 2
    xy = tf.unstack(sample_coords, axis=len(sample_coords.get_shape()) - 1)
    index_voxel_coords = [tf.floor(x) for x in xy]

    def boundary_replicate(sample_coords0, input_size0):

        return tf.maximum(tf.minimum(sample_coords0, input_size0 - 1), 0)

    spatial_coords = [boundary_replicate(tf.cast(x, tf.int32), input_size[idx])
                      for idx, x in enumerate(index_voxel_coords)]

    spatial_coords_plus1 = [boundary_replicate(tf.cast(x + 1., tf.int32), input_size[idx])
                            for idx, x in enumerate(index_voxel_coords)]

    weight = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for x, i in zip(xy, spatial_coords)]
    weight_c = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for x, i in zip(xy, spatial_coords_plus1)]

    sz = spatial_coords[0].get_shape().as_list()
    batch_coords = tf.tile(tf.reshape(tf.range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)),
                           [1] + sz[1:])
    sc = (spatial_coords, spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i, '0%ib' % spatial_rank)] for i in range(2 ** spatial_rank)]

    make_sample = lambda bc: tf.gather_nd(inputs, tf.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1))

    samples = [make_sample(bc) for bc in binary_codes]


    #    def pyramid_combination(samples0, weight0, weight_c0):
    #        if len(weight0) == 1:
    #            return samples0[0]*weight_c0[0]+samples0[1]*weight0[0]
    #        else:
    #            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
    #                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]
    #
    #    return pyramid_combination(samples, weight, weight_c)
    #    weight_c = [tf.ones_like(weight_c[0]), tf.ones_like(weight_c[1]), tf.ones_like(weight_c[2])]
    #    weight = [tf.zeros_like(weight[0]), tf.zeros_like(weight[1]), tf.zeros_like(weight[2])]
    def pyramid_combination(samples0, weight0, weight_c0):
        if len(weight0) == 1:
            return samples0[0] * weight_c0[0] + samples0[1] * weight0[0]
        else:
            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]

    return pyramid_combination(samples, weight, weight_c)


def get_reference_grid(grid_size):
    return tf.cast(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3), dtype=tf.float32)


def get_reference_grid1(grid_size):
    return np.stack(np.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3)



def compute_binary_dice(input1, input2):
    # input1 = tf.pad(input1, [[0, 0], [0, 16], [0, 0], [0, 8], [0, 0]])

    mask1 = input1 >= 0.5
    mask2 = input2 >= 0.5
    vol1 = tf.reduce_sum(tf.compat.v1.to_float(mask1), axis=[1, 2, 3, 4])
    vol2 = tf.reduce_sum(tf.compat.v1.to_float(mask2), axis=[1, 2, 3, 4])
    dice = tf.reduce_sum(tf.compat.v1.to_float(mask1 & mask2), axis=[1, 2, 3, 4]) * 2 / (vol1 + vol2)
    return dice


def compute_centroid_distance(input1, input2, grid=None):
    # input1 = tf.pad(input1, [[0, 0], [0, 16], [0, 0], [0, 8], [0, 0]])

    if grid is None:
        grid = get_reference_grid(input1.get_shape()[1:4])

    def compute_centroid(mask, grid0):

        return tf.stack([tf.reduce_mean(tf.boolean_mask(grid0, mask[i, ..., 0] >= 0.5), axis=0)
                         for i in range(mask.shape[0])], axis=0)
        # bool = tf.boolean_mask(grid0, mask[0, ..., 0] >= 0.5)
        # mean = tf.reduce_mean(bool, axis=0)
        # return tf.stack(mean, axis=0)

    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1 - c2), axis=1))


def flip_image_left_right(input_):
    # image = tf.reverse(input_, axis=1)
    image = tf.reverse(input_, [1])
    return image


def flip_random_image(input_):
    # value = tf.random_uniform([1], 0, 1)
    # rule = tf.constant(0.5)
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        input_ = tf.squeeze(input_)
        input_moving_image1, input_moving_image2, input_fixed_image1, input_fixed_image2, input_moving_label1, \
        input_moving_label2, input_fixed_label1, input_fixed_label2 = tf.split(input_, num_or_size_splits=8, axis=0)
        input_moving_image1 = tf.expand_dims(flip_image_left_right(input_moving_image1), 4)
        input_moving_image2 = tf.expand_dims(flip_image_left_right(input_moving_image2), 4)
        input_fixed_image1 = tf.expand_dims(flip_image_left_right(input_fixed_image1), 4)
        input_fixed_image2 = tf.expand_dims(flip_image_left_right(input_fixed_image2), 4)
        input_moving_label1 = tf.expand_dims(flip_image_left_right(input_moving_label1), 4)
        input_moving_label2 = tf.expand_dims(flip_image_left_right(input_moving_label2), 4)
        input_fixed_label1 = tf.expand_dims(flip_image_left_right(input_fixed_label1), 4)
        input_fixed_label2 = tf.expand_dims(flip_image_left_right(input_fixed_label2), 4)
        return tf.concat([input_moving_image1, input_moving_image2], axis=0), tf.concat([input_fixed_image1,
                                                                                         input_fixed_image2],
                                                                                        axis=0), tf.concat(
            [input_moving_label1, input_moving_label2], axis=0), \
               tf.concat([input_fixed_label1, input_fixed_label2], axis=0)
    else:

        input_moving_image, input_fixed_image, input_moving_label, input_fixed_label = tf.split(input_, 4)
        return input_moving_image, input_fixed_image, input_moving_label, input_fixed_label



