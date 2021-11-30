import os
import nibabel as nib
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
data_path_T1GD_label = r'/data/jdm_data/else/cross1/test/T1GD_labels'

data_path_TOF_label = r'/data/jdm_data/else/cross1/save'
# data_path_T1GD_label = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\cross1\test\T1GD_labels'
# data_path_TOF_label = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\cross1\test\TOF_labels'
filename_T1GD_label0 = os.path.join(data_path_T1GD_label, 'case000001.nii.gz')
filename_T1GD_label1 = os.path.join(data_path_T1GD_label, 'case000003.nii.gz')
filename_T1GD_label2 = os.path.join(data_path_T1GD_label, 'case000008.nii.gz')
filename_T1GD_label3 = os.path.join(data_path_T1GD_label, 'case000013.nii.gz')
filename_T1GD_label4 = os.path.join(data_path_T1GD_label, 'case000015.nii.gz')
filename_TOF_label0 = os.path.join(data_path_TOF_label, 'warped_label0.nii.gz')
filename_TOF_label1 = os.path.join(data_path_TOF_label, 'warped_label1.nii.gz')
filename_TOF_label2 = os.path.join(data_path_TOF_label, 'warped_label2.nii.gz')
filename_TOF_label3 = os.path.join(data_path_TOF_label, 'warped_label3.nii.gz')
filename_TOF_label4 = os.path.join(data_path_TOF_label, 'warped_label4.nii.gz')
# filename_TOF_label0 = os.path.join(data_path_TOF_label, 'case000001.nii.gz')
# filename_TOF_label1 = os.path.join(data_path_TOF_label, 'case000003.nii.gz')
# filename_TOF_label2 = os.path.join(data_path_TOF_label, 'case000008.nii.gz')
# filename_TOF_label3 = os.path.join(data_path_TOF_label, 'case000013.nii.gz')
# filename_TOF_label4 = os.path.join(data_path_TOF_label, 'case000015.nii.gz')


# T1GD_label = nib.load(filename_T1GD_label)
# TOF_label = nib.load(filename_TOF_label)
# T1GD_label_data = T1GD_label.get_data()
# T1GD_label_data = T1GD_label.get_data()
# TOF_label_data = TOF_label.get_data()

# TOF_label_data = np.flip(np.transpose(TOF_label_data, [0, 2, 1]), 1)
# T1GD_label_get_data = tf.to_float(tf.convert_to_tensor(T1GD_label_data))
# T1GD_label_get_data = tf.to_float(tf.convert_to_tensor(T1GD_label_data))
# TOF_label_get_data = tf.squeeze(tf.to_float(tf.convert_to_tensor(TOF_label_data)))
# t1gd_data = tf.convert_to_tensor(np.asarray(T1GD_label_data), dtype=tf.float32)
# tof_data = tf.convert_to_tensor(np.asarray(np.flip(np.transpose(TOF_label_data, (0, 2, 1)), 1)), dtype=tf.float32)

# def compute_binary_dice(input1, input2):
#     mask1 = input1 >= 0.5
#     mask2 = input2 >= 0.5
#     vol1 = tf.reduce_sum(tf.to_float(mask1), axis=[0, 1, 2])
#     vol2 = tf.reduce_sum(tf.to_float(mask2), axis=[0, 1, 2])

#     dice = tf.reduce_sum(tf.to_float(mask1 & mask2), axis=[0, 1, 2])*2 / (vol1+vol2)
#     return dice
def label_information(file_t1gd, file_tof):
    T1GD_label = nib.load(file_t1gd)
    TOF_label = nib.load(file_tof)
    T1GD_label_data = T1GD_label.get_data()
    TOF_label_data = np.squeeze(TOF_label.get_data())
    t1gd_data = tf.convert_to_tensor(np.asarray(T1GD_label_data), dtype=tf.float32)
    tof_data = tf.convert_to_tensor(np.asarray(np.flip(np.transpose(TOF_label_data, (0, 2, 1)), 1)), dtype=tf.float32)
    return t1gd_data, tof_data


def dice_simple(ts, ps, eps_vol=1e-6):
    # ps = tf.pad(ps, [[0, 16], [0, 0], [0, 8]])
    numerator = tf.reduce_sum(ts * ps, axis=[0, 1, 2]) * 2
    denominator = tf.reduce_sum(ts, axis=[0, 1, 2]) + tf.reduce_sum(ps, axis=[0, 1, 2]) + eps_vol
    return numerator / denominator


def compute_centroid_distance(input1, input2, grid=None):
    # input2 = tf.pad(input2, [[0, 16], [0, 0], [0, 8]])
    if grid is None:
        grid = get_reference_grid(input1.get_shape()[0:3])


    def compute_centroid(mask, grid0):

        # return tf.reduce_mean(tf.boolean_mask(grid0, mask >= 0.5), axis=0)
        a = tf.boolean_mask(grid0, mask >= 0.5)
        b= tf.reduce_mean(a, axis=0)
        return b

    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1 - c2), axis=0))


def get_reference_grid(grid_size):
    return tf.cast(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3), dtype=tf.float32)



t1gd0_data, tof0_data = label_information(filename_T1GD_label0, filename_TOF_label0)
t1gd1_data, tof1_data = label_information(filename_T1GD_label1, filename_TOF_label1)
t1gd2_data, tof2_data = label_information(filename_T1GD_label2, filename_TOF_label2)
t1gd3_data, tof3_data = label_information(filename_T1GD_label3, filename_TOF_label3)
t1gd4_data, tof4_data = label_information(filename_T1GD_label4, filename_TOF_label4)

dice0 = dice_simple(t1gd0_data, tof0_data)
dice1 = dice_simple(t1gd1_data, tof1_data)
dice2 = dice_simple(t1gd2_data, tof2_data)
dice3 = dice_simple(t1gd3_data, tof3_data)
dice4 = dice_simple(t1gd4_data, tof4_data)

centroid_distance0 = compute_centroid_distance(t1gd0_data, tof0_data)
centroid_distance1 = compute_centroid_distance(t1gd1_data, tof1_data)
centroid_distance2 = compute_centroid_distance(t1gd2_data, tof2_data)
centroid_distance3 = compute_centroid_distance(t1gd3_data, tof3_data)
centroid_distance4 = compute_centroid_distance(t1gd4_data, tof4_data)

sess = tf.compat.v1.InteractiveSession()

dice0 = round(dice0.eval(), 3)
dice1 = round(dice1.eval(), 3)
dice2 = round(dice2.eval(), 3)
dice3 = round(dice3.eval(), 3)
dice4 = round(dice4.eval(), 3)
centroid_distance0 = round(centroid_distance0.eval(), 3)
centroid_distance1 = round(centroid_distance1.eval(), 3)
centroid_distance2 = round(centroid_distance2.eval(), 3)
centroid_distance3 = round(centroid_distance3.eval(), 3)
centroid_distance4 = round(centroid_distance4.eval(), 3)
print("dice", dice0, dice1, dice2, dice3, dice4,)
print("mean dice", round((dice0+dice1+dice2+dice3+dice4)/5, 3))
print("centroid distance", centroid_distance0, centroid_distance1, centroid_distance2, centroid_distance3,centroid_distance4,)
print("mean centroid distance", round((centroid_distance0+centroid_distance1+centroid_distance2+centroid_distance3+centroid_distance4)/5, 3))
