import nibabel as nib
import os
import numpy as np


data_path_T1GD_image = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\train\T1GD_image'
data_path_T1GD_label = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\train\T1GD_label'
data_path_TOF_image = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\train\TOF_image'
data_path_TOF_label = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\train\TOF_label'
filename_T1GD_image = os.path.join(data_path_T1GD_image, 'case000000.nii.gz')
filename_T1GD_label = os.path.join(data_path_T1GD_label, 'case000000.nii.gz')
filename_TOF_image = os.path.join(data_path_TOF_image, 'case000000.nii.gz')
filename_TOF_label = os.path.join(data_path_TOF_label, 'case000000.nii.gz')


img_T1GD = nib.load(filename_T1GD_image)
img_TOF = nib.load(filename_TOF_image)
label_T1GD = nib.load(filename_T1GD_label)
label_TOF = nib.load(filename_TOF_label)
img_T1GD_get_data = img_T1GD.get_data()
img_T1GD_dataobj = img_T1GD.dataobj[:, :, :]
img_TOF_get_data = img_TOF.get_data()
label_T1GD_get_data = label_T1GD.get_data()
label_TOF_get_data = label_TOF.get_data()
transpose_img_tof_data = np.transpose(img_TOF_get_data, [0, 2, 1])
mirr_img_tof_data = np.flip(transpose_img_tof_data, 1)
print('show')

