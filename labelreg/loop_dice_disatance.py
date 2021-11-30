import os
import nibabel as nib
import tensorflow as tf
import numpy as np
import pandas as pd

def label_information(file_t1ce, file_t2):
    t1ce_label = nib.load(file_t1ce)
    t2_label = nib.load(file_t2)
    t1ce_label_data = t1ce_label.get_data()
    t2_label_data = np.squeeze(t2_label.get_data())
    t1ce_data = tf.convert_to_tensor(np.asarray(t1ce_label_data), dtype=tf.float32)
    t2_data = tf.convert_to_tensor(np.asarray(t2_label_data), dtype=tf.float32)
    return t1ce_data, t2_data


def dice_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts * ps, axis=[0, 1, 2]) * 2
    denominator = tf.reduce_sum(ts, axis=[0, 1, 2]) + tf.reduce_sum(ps, axis=[0, 1, 2]) + eps_vol
    return numerator / denominator


def compute_centroid_distance(input1, input2, grid=None):
    if grid is None:
        grid = get_reference_grid(input1.get_shape()[0:3])

    def compute_centroid(mask, grid0):

        # return tf.reduce_mean(tf.boolean_mask(grid0, mask >= 0.5), axis=0)
        a = tf.boolean_mask(grid0, mask >= 0.5)
        b = tf.reduce_mean(a, axis=0)
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


def dirpath(lpath, lfilelist):
    list = os.listdir(lpath)
    for f in list:
        file = os.path.join(lpath, f)
        if os.path.isdir(file):
            dirpath(file, lfilelist)
        else:
            lfilelist.append(file)
    return lfilelist


def dirpath_t2name(lpath, lfilelist):
    list = os.listdir(lpath)
    for f in list:
        file = os.path.join(lpath, f)
        if os.path.isdir(file):
            dirpath(file, lfilelist)
        else:
            lfilelist.append(file)
    sparce = []
    for f_T1CE in lfilelist:
        sparce.append(f_T1CE[-20:])
    return sparce


def dirpath_t1cename(lpath, lfilelist):
    list = os.listdir(lpath)
    for f in list:
        file = os.path.join(lpath, f)
        if os.path.isdir(file):
            dirpath(file, lfilelist)
        else:
            lfilelist.append(file)
    sparce = []
    for f_T1CE in lfilelist:
        sparce.append(f_T1CE[-17:])
    return sparce


def listdata(data_path_t1ce, lfilelist_t1ce, data_path_t2, lfilelist_t2):
    sparse_t1ce = []
    sparse_t2 = []
    for i in lfilelist_t1ce:
        sparse_t1ce.append(os.path.join(data_path_t1ce, i))
    for i in lfilelist_t2:
        sparse_t2.append(os.path.join(data_path_t2, i))
    return sparse_t1ce, sparse_t2


data_path_T1CE_label = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\BraTS_data\cross1\test1\T1CE_labels'
data_path_T2_label = r'C:\Users\Administrator\Desktop\save\test1'


T1CE_label_lfilelist = dirpath_t1cename(data_path_T1CE_label, [])
T2_label_lfilelist = dirpath_t2name(data_path_T2_label, [])

filename_t1ce_list, filename_t2_list = listdata(data_path_T1CE_label, T1CE_label_lfilelist, data_path_T2_label, T2_label_lfilelist)

t1ce0_data, t20_date = label_information(filename_t1ce_list[0], filename_t2_list[20])
dice0 = dice_simple(t1ce0_data, t20_date)
centroid_distance0 = compute_centroid_distance(t1ce0_data, t20_date)

t1ce1_data, t21_date = label_information(filename_t1ce_list[1], filename_t2_list[21])
dice1 = dice_simple(t1ce1_data, t21_date)
centroid_distance1 = compute_centroid_distance(t1ce1_data, t21_date)

t1ce2_data, t22_date = label_information(filename_t1ce_list[2], filename_t2_list[22])
dice2 = dice_simple(t1ce2_data, t22_date)
centroid_distance2 = compute_centroid_distance(t1ce2_data, t22_date)

t1ce3_data, t23_date = label_information(filename_t1ce_list[3], filename_t2_list[23])
dice3 = dice_simple(t1ce3_data, t23_date)
centroid_distance3 = compute_centroid_distance(t1ce3_data, t23_date)

t1ce4_data, t24_date = label_information(filename_t1ce_list[4], filename_t2_list[24])
dice4 = dice_simple(t1ce4_data, t24_date)
centroid_distance4 = compute_centroid_distance(t1ce4_data, t24_date)

t1ce5_data, t25_date = label_information(filename_t1ce_list[5], filename_t2_list[25])
dice5 = dice_simple(t1ce5_data, t25_date)
centroid_distance5 = compute_centroid_distance(t1ce5_data, t25_date)

t1ce6_data, t26_date = label_information(filename_t1ce_list[6], filename_t2_list[26])
dice6 = dice_simple(t1ce6_data, t26_date)
centroid_distance6 = compute_centroid_distance(t1ce6_data, t26_date)

t1ce7_data, t27_date = label_information(filename_t1ce_list[7], filename_t2_list[27])
dice7 = dice_simple(t1ce7_data, t27_date)
centroid_distance7 = compute_centroid_distance(t1ce7_data, t27_date)

t1ce8_data, t28_date = label_information(filename_t1ce_list[8], filename_t2_list[28])
dice8 = dice_simple(t1ce8_data, t28_date)
centroid_distance8 = compute_centroid_distance(t1ce8_data, t28_date)

t1ce9_data, t29_date = label_information(filename_t1ce_list[9], filename_t2_list[29])
dice9 = dice_simple(t1ce9_data, t29_date)
centroid_distance9 = compute_centroid_distance(t1ce9_data, t29_date)


sess = tf.InteractiveSession()
dice0 = round(dice0.eval(), 3)
dice1 = round(dice1.eval(), 3)
dice2 = round(dice2.eval(), 3)
dice3 = round(dice3.eval(), 3)
dice4 = round(dice4.eval(), 3)
dice5 = round(dice5.eval(), 3)
dice6 = round(dice6.eval(), 3)
dice7 = round(dice7.eval(), 3)
dice8 = round(dice8.eval(), 3)
dice9 = round(dice9.eval(), 3)


centroid_distance0 = round(centroid_distance0.eval(), 3)
centroid_distance1 = round(centroid_distance1.eval(), 3)
centroid_distance2 = round(centroid_distance2.eval(), 3)
centroid_distance3 = round(centroid_distance3.eval(), 3)
centroid_distance4 = round(centroid_distance4.eval(), 3)
centroid_distance5 = round(centroid_distance5.eval(), 3)
centroid_distance6 = round(centroid_distance6.eval(), 3)
centroid_distance7 = round(centroid_distance7.eval(), 3)
centroid_distance8 = round(centroid_distance8.eval(), 3)
centroid_distance9 = round(centroid_distance9.eval(), 3)

data_list_dice = [dice0, dice1, dice2, dice3, dice4, dice5, dice6, dice7, dice8, dice9]
data_list_distance = [centroid_distance0, centroid_distance1, centroid_distance2\
 , centroid_distance3, centroid_distance4,centroid_distance5, centroid_distance6, centroid_distance7, centroid_distance8, centroid_distance9]
test_match = pd.DataFrame({'dice':data_list_dice, 'distance':data_list_distance})
test_match.to_csv('C:/Users/Administrator/Desktop/test2.csv')


print("dice：", dice0, dice1, dice2, dice3, dice4, dice5, dice6, dice7, dice8, dice9)
print("mean dice", round((dice0+dice1+dice2+dice3+dice4+dice5+dice6+dice7+dice8+dice9)/10, 3))
print("centroid distance ", centroid_distance0, centroid_distance1, centroid_distance2, centroid_distance3,centroid_distance4,\
      centroid_distance5, centroid_distance6, centroid_distance7, centroid_distance8, centroid_distance9)
print("mean centroid distance ", round((centroid_distance0+centroid_distance1+centroid_distance2+centroid_distance3+\
    centroid_distance4+centroid_distance5+centroid_distance6+centroid_distance7+centroid_distance8+centroid_distance9)/10, 3))

