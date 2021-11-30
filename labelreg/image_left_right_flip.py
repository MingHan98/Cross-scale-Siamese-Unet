import os
import nibabel as nib
import numpy as np

data_path = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\jdm_data\cross1\train\TOF_images'
save_path = r"C:\Users\Administrator\Desktop\reg_tutorials\code_data\jdm_data\cross1\train\TOF_images"
filename_T1GD_label0 = os.path.join(data_path, 'case000000.nii.gz')
filename_T1GD_label1 = os.path.join(data_path, 'case000002.nii.gz')
filename_T1GD_label2 = os.path.join(data_path, 'case000006.nii.gz')
filename_T1GD_label3 = os.path.join(data_path, 'case000007.nii.gz')
filename_T1GD_label4 = os.path.join(data_path, 'case000009.nii.gz')
filename_T1GD_label5 = os.path.join(data_path, 'case000010.nii.gz')
filename_T1GD_label6 = os.path.join(data_path, 'case000012.nii.gz')
filename_T1GD_label7 = os.path.join(data_path, 'case000016.nii.gz')
filename_T1GD_label8 = os.path.join(data_path, 'case000017.nii.gz')
filename_T1GD_label9 = os.path.join(data_path, 'case000018.nii.gz')
filename_T1GD_label10 = os.path.join(data_path, 'case000019.nii.gz')
filename_T1GD_label11 = os.path.join(data_path, 'case000020.nii.gz')
filename_T1GD_label12 = os.path.join(data_path, 'case000021.nii.gz')
filename_T1GD_label13 = os.path.join(data_path, 'case000022.nii.gz')
filename_T1GD_label14 = os.path.join(data_path, 'case000023.nii.gz')


def t1gd_label_information(file_t1gd):
    T1GD_label = nib.load(file_t1gd)
    T1GD_label_data = T1GD_label.get_data()
    T1GD_labe_affine = T1GD_label.affine.copy()
    T1GD_labe_hdr = T1GD_label.header.copy()
#   T1GD_label_data = np.flip(T1GD_label_data, axis=0)
    T1GD_label_data = T1GD_label_data[::-1, :, :]

    return T1GD_label_data, T1GD_labe_affine, T1GD_labe_hdr


# def save_niftimage(data, affine, hdr):
#     save_T1GD_label = nib.Nifti1Image(data, affine, hdr)
#     return save_T1GD_label


T1GD_label0_data, T1GD_label0_affine, T1GD_label0_hdr = t1gd_label_information(filename_T1GD_label0)
T1GD_label1_data, T1GD_label1_affine, T1GD_label1_hdr = t1gd_label_information(filename_T1GD_label1)
T1GD_label2_data, T1GD_label2_affine, T1GD_label2_hdr = t1gd_label_information(filename_T1GD_label2)
T1GD_label3_data, T1GD_label3_affine, T1GD_label3_hdr = t1gd_label_information(filename_T1GD_label3)
T1GD_label4_data, T1GD_label4_affine, T1GD_label4_hdr = t1gd_label_information(filename_T1GD_label4)
T1GD_label5_data, T1GD_label5_affine, T1GD_label5_hdr = t1gd_label_information(filename_T1GD_label5)
T1GD_label6_data, T1GD_label6_affine, T1GD_label6_hdr = t1gd_label_information(filename_T1GD_label6)
T1GD_label7_data, T1GD_label7_affine, T1GD_label7_hdr = t1gd_label_information(filename_T1GD_label7)
T1GD_label8_data, T1GD_label8_affine, T1GD_label8_hdr = t1gd_label_information(filename_T1GD_label8)
T1GD_label9_data, T1GD_label9_affine, T1GD_label9_hdr = t1gd_label_information(filename_T1GD_label9)
T1GD_label10_data, T1GD_label10_affine, T1GD_label10_hdr = t1gd_label_information(filename_T1GD_label10)
T1GD_label11_data, T1GD_label11_affine, T1GD_label11_hdr = t1gd_label_information(filename_T1GD_label11)
T1GD_label12_data, T1GD_label12_affine, T1GD_label12_hdr = t1gd_label_information(filename_T1GD_label12)
T1GD_label13_data, T1GD_label13_affine, T1GD_label13_hdr = t1gd_label_information(filename_T1GD_label13)
T1GD_label14_data, T1GD_label14_affine, T1GD_label14_hdr = t1gd_label_information(filename_T1GD_label14)

save_T1GD_labe0 = nib.Nifti1Image(T1GD_label0_data, T1GD_label0_affine, T1GD_label0_hdr)
save_T1GD_labe1 = nib.Nifti1Image(T1GD_label1_data, T1GD_label1_affine, T1GD_label1_hdr)
save_T1GD_labe2 = nib.Nifti1Image(T1GD_label2_data, T1GD_label2_affine, T1GD_label2_hdr)
save_T1GD_labe3 = nib.Nifti1Image(T1GD_label3_data, T1GD_label3_affine, T1GD_label3_hdr)
save_T1GD_labe4 = nib.Nifti1Image(T1GD_label4_data, T1GD_label4_affine, T1GD_label4_hdr)
save_T1GD_labe5 = nib.Nifti1Image(T1GD_label5_data, T1GD_label5_affine, T1GD_label5_hdr)
save_T1GD_labe6 = nib.Nifti1Image(T1GD_label6_data, T1GD_label6_affine, T1GD_label6_hdr)
save_T1GD_labe7 = nib.Nifti1Image(T1GD_label7_data, T1GD_label7_affine, T1GD_label7_hdr)
save_T1GD_labe8 = nib.Nifti1Image(T1GD_label8_data, T1GD_label8_affine, T1GD_label8_hdr)
save_T1GD_labe9 = nib.Nifti1Image(T1GD_label9_data, T1GD_label9_affine, T1GD_label9_hdr)
save_T1GD_labe10 = nib.Nifti1Image(T1GD_label10_data, T1GD_label10_affine, T1GD_label10_hdr)
save_T1GD_labe11 = nib.Nifti1Image(T1GD_label11_data, T1GD_label11_affine, T1GD_label11_hdr)
save_T1GD_labe12 = nib.Nifti1Image(T1GD_label12_data, T1GD_label12_affine, T1GD_label12_hdr)
save_T1GD_labe13 = nib.Nifti1Image(T1GD_label13_data, T1GD_label13_affine, T1GD_label13_hdr)
save_T1GD_labe14 = nib.Nifti1Image(T1GD_label14_data, T1GD_label14_affine, T1GD_label14_hdr)
nib.save(save_T1GD_labe0, os.path.join(save_path, 'case000024.nii.gz'))
nib.save(save_T1GD_labe1, os.path.join(save_path, 'case000025.nii.gz'))
nib.save(save_T1GD_labe2, os.path.join(save_path, 'case000026.nii.gz'))
nib.save(save_T1GD_labe3, os.path.join(save_path, 'case000027.nii.gz'))
nib.save(save_T1GD_labe4, os.path.join(save_path, 'case000028.nii.gz'))
nib.save(save_T1GD_labe5, os.path.join(save_path, 'case000029.nii.gz'))
nib.save(save_T1GD_labe6, os.path.join(save_path, 'case000030.nii.gz'))
nib.save(save_T1GD_labe7, os.path.join(save_path, 'case000031.nii.gz'))
nib.save(save_T1GD_labe8, os.path.join(save_path, 'case000032.nii.gz'))
nib.save(save_T1GD_labe9, os.path.join(save_path, 'case000033.nii.gz'))
nib.save(save_T1GD_labe10, os.path.join(save_path, 'case000034.nii.gz'))
nib.save(save_T1GD_labe11, os.path.join(save_path, 'case000035.nii.gz'))
nib.save(save_T1GD_labe12, os.path.join(save_path, 'case000036.nii.gz'))
nib.save(save_T1GD_labe13, os.path.join(save_path, 'case000037.nii.gz'))
nib.save(save_T1GD_labe14, os.path.join(save_path, 'case000038.nii.gz'))
print('over')
