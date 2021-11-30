import nibabel as nib
import os


data_path_T1GD = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\T1GD_left_right_up_down\17'
data_path_TOF = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\TOF_left_right_front_back\17'
filename_T1GD_left = os.path.join(data_path_T1GD, 'T1GD_left_ud.nii.gz')
filename_T1GD_right = os.path.join(data_path_T1GD, 'T1GD_right_ud.nii.gz')
filename_TOF_left = os.path.join(data_path_TOF, 'TOF_left_fb.nii.gz')
filename_TOF_right = os.path.join(data_path_TOF, 'TOF_right_fb.nii.gz')

save_path_T1GD_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\ks\17\T1GD_left.nii.gz'
save_path_T1GD_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\ks\17\T1GD_right.nii.gz'
save_path_TOF_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\ks\17\TOF_left.nii.gz'
save_path_TOF_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\ks\17\TOF_right.nii.gz'

img_T1GD_left = nib.load(filename_T1GD_left)
img_TOF_left = nib.load(filename_TOF_left)
img_T1GD_right = nib.load(filename_T1GD_right)
img_TOF_right = nib.load(filename_TOF_right)
img_T1GD_left_get_data = img_T1GD_left.get_data()
img_T1GD_right_get_data = img_T1GD_right.get_data()
img_TOF_left_get_data = img_TOF_left.get_data()
img_TOF_right_get_data = img_TOF_right.get_data()
affine_left_T1GD = img_T1GD_left.affine.copy()
affine_right_T1GD = img_T1GD_right.affine.copy()
affine_left_TOF = img_TOF_left.affine.copy()
affine_right_TOF = img_TOF_right.affine.copy()


affine_left_T1GD[0][0], affine_left_T1GD[2][1], affine_left_T1GD[1][2] = -0.6, -0.6, -0.6
affine_right_T1GD[0][0], affine_right_T1GD[2][1], affine_right_T1GD[1][2] = -0.6, -0.6, -0.6
affine_left_TOF[0][0], affine_left_TOF[1][1], affine_left_TOF[2][2] = -0.6, -0.6, 0.6
affine_right_TOF[0][0], affine_right_TOF[1][1], affine_right_TOF[2][2] = -0.6, -0.6, 0.6


nib.save(nib.Nifti1Image(img_T1GD_left_get_data[:, :, :], affine=affine_left_T1GD), save_path_T1GD_left)
nib.save(nib.Nifti1Image(img_T1GD_right_get_data[:, :, :], affine=affine_right_T1GD), save_path_T1GD_right)
nib.save(nib.Nifti1Image(img_TOF_left_get_data[:, :, :], affine=affine_left_TOF), save_path_TOF_left)
nib.save(nib.Nifti1Image(img_TOF_right_get_data[:, :, :], affine=affine_right_TOF), save_path_TOF_right)

print('finish')

