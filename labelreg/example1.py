import tensorflow as tf
import numpy as np
import nibabel as nib
import os

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


def gauss_kernel1d(sigma):
    if sigma == 0:
        return 0
    else:
        tail = int(sigma * 3)
        k = tf.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])

        return k / tf.reduce_sum(k)

# data_path_T1GD_image = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\train\T1GD_image'
data_path_T1GD_label = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\jdm_data\cross1\train\T1GD_labels'
# data_path_TOF_image = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\train\TOF_image'
# data_path_TOF_label = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\train\TOF_label'
# filename_T1GD_image = os.path.join(data_path_T1GD_image, 'case000000.nii.gz')
filename_T1GD_label = os.path.join(data_path_T1GD_label, 'case000007.nii.gz')
# filename_TOF_image = os.path.join(data_path_TOF_image, 'case000000.nii.gz')
# filename_TOF_label = os.path.join(data_path_TOF_label, 'case000000.nii.gz')
save_path = r"C:\Users\Administrator\Desktop\label_test"



# img_T1GD = nib.load(filename_T1GD_image)
# img_TOF = nib.load(filename_TOF_image)
label_T1GD = nib.load(filename_T1GD_label)
# label_TOF = nib.load(filename_TOF_label)
# img_T1GD_get_data = img_T1GD.get_data()
# img_T1GD_dataobj = img_T1GD.dataobj[:, :, :]
# img_TOF_get_data = img_TOF.get_data()
img_test_affine = label_T1GD.affine.copy()
img_test_hdr = label_T1GD.header.copy()
label_T1GD_get_data = label_T1GD.get_data()
label_T1GD_get_data= np.expand_dims(label_T1GD_get_data, axis=0)
label_T1GD_get_data= np.expand_dims(label_T1GD_get_data, axis=-1)
# label_TOF_get_data = label_TOF.get_data()
# transpose_img_tof_data = np.transpose(img_TOF_get_data, [0, 2, 1])
# mirr_img_tof_data = np.flip(transpose_img_tof_data, 1)

t = tf.convert_to_tensor(label_T1GD_get_data, tf.float32, name='t')
GK = gauss_kernel1d(16)
GK_SF = separable_filter3d(t, GK)
GK_SF= tf.squeeze(GK_SF)


sess = tf.Session()
img_numpy = GK_SF.eval(session = sess)

save_warped_label4 = nib.Nifti1Image(img_numpy, img_test_affine, img_test_hdr)
nib.save(save_warped_label4, os.path.join(save_path, 'test_label16.nii.gz'))

print('ok')