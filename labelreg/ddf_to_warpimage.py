import labelreg.apps as app
import nibabel as nib
import numpy as np
import os
import tensorflow as tf
move_label_path = r'C:\Users\Administrator\Desktop\manual_registration\23'
fixed_label_path = r'C:\Users\Administrator\Desktop\manual_registration\23'
ddf_path = r'C:\Users\Administrator\Desktop\manual_registration\23'
save_path = r'C:\Users\Administrator\Desktop\manual_registration\23'

ddf = os.path.join(move_label_path, 'ddf.nii.gz')
move_label = os.path.join(move_label_path, 'move_label.nii.gz')
fixed_label = os.path.join(fixed_label_path, 'fixed_label.nii.gz')
ddf_image = nib.load(ddf)
move_label_image = nib.load(move_label)

move_label_data = move_label_image.get_data()
move_label_affine = move_label_image.affine.copy()
move_label_hdr = move_label_image.header.copy()
ddf_data = ddf_image.get_data()
ddf_affine = ddf_image.affine.copy()
ddf_hdr = ddf_image.header.copy()
fixed_label_data = move_label_image.get_data()



ddf_data = np.expand_dims(np.squeeze(ddf_image.get_data()), 0)
# move_label_data = np.expand_dims(np.expand_dims(np.flip(np.transpose(move_label_image.get_data(), [0, 2, 1]), 1), 0), -1)
move_label_data = np.expand_dims(np.expand_dims(move_label_image.get_data(), 0), -1)
warped_labels = np.squeeze(app.warp_volumes_by_ddf(move_label_data, ddf_data))

def dice_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts * ps, axis=[0, 1, 2]) * 2
    denominator = tf.reduce_sum(ts, axis=[0, 1, 2]) + tf.reduce_sum(ps, axis=[0, 1, 2]) + eps_vol
    return numerator / denominator


def compute_centroid_distance(input1, input2, grid=None):
    if grid is None:
        grid = get_reference_grid(input1.get_shape()[0:3])

    def compute_centroid(mask, grid0):

        return tf.reduce_mean(tf.boolean_mask(grid0, mask >= 0.5), axis=0)

    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1 - c2), axis=0))


def get_reference_grid(grid_size):
    return tf.cast(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3), dtype=tf.float32)

warped_labels = tf.convert_to_tensor(warped_labels, dtype=tf.float32)
fixed_label_data = tf.convert_to_tensor(fixed_label_data, dtype=tf.float32)

dice = dice_simple(fixed_label_data, warped_labels)
centroid_distance = compute_centroid_distance(fixed_label_data, warped_labels)
sess = tf.InteractiveSession()
dice = round(dice.eval(), 3)
centroid_distance = round(centroid_distance.eval(), 3)
print("dice：", dice)
print("centroid distance", centroid_distance)