import nibabel as nib
import os


data_path_T1GD = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\T1GD_left_right_up_down\31'
data_path_TOF = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\TOF_left_right_front_back\31'

filename_T1GD_left = os.path.join(data_path_T1GD, 'T1GD_left_ud.nii.gz')
filename_T1GD_right = os.path.join(data_path_T1GD, 'T1GD_right_ud.nii.gz')
filename_TOF_left = os.path.join(data_path_TOF, 'TOF_left_fb.nii.gz')
filename_TOF_right = os.path.join(data_path_TOF, 'TOF_right_fb.nii.gz')
save_T1GD_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\up_down_slicer\31\T1GD_left.nii.gz'
save_T1GD_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\up_down_slicer\31\T1GD_right.nii.gz'
save_TOF_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\up_down_slicer\31\TOF_left.nii.gz'
save_TOF_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\up_down_slicer\31\TOF_right.nii.gz'

img_T1GD_left = nib.load(filename_T1GD_left)
img_T1GD_right = nib.load(filename_T1GD_right)
img_TOF_left = nib.load(filename_TOF_left)
img_TOF_right = nib.load(filename_TOF_right)
img_T1GD_left_get_data = img_T1GD_left.get_data()
img_T1GD_right_get_data = img_T1GD_right.get_data()
img_TOF_left_get_data = img_TOF_left.get_data()
img_TOF_right_get_data = img_TOF_right.get_data()
hdr_T1GD_left = img_T1GD_left.header.copy()
hdr_T1GD_right = img_T1GD_right.header.copy()
hdr_TOF_left = img_TOF_left.header.copy()
hdr_TOF_right = img_TOF_right.header.copy()
affine_T1GD_left = img_T1GD_left.affine.copy()
affine_T1GD_right = img_T1GD_right.affine.copy()
affine_TOF_left = img_TOF_left.affine.copy()
affine_TOF_right = img_TOF_right.affine.copy()


def t1gd_up_down_cat(input_image):
    data = input_image.get_data()
    hdr = input_image.header.copy()
    affine = input_image.affine.copy()
    output = nib.Nifti1Image(data[:, data.shape[1] - 166:data.shape[1] - 26, :], affine=affine, header=hdr)
    return output


def tof_up_down_cat(input_image):
    data = input_image.get_data()
    hdr = input_image.header.copy()
    affine = input_image.affine.copy()
    correct = 63
    affine[2][3] = affine[2][2] * correct + affine[2][3]
    output = nib.Nifti1Image(data[:, :, data.shape[2] - 181:data.shape[2] - 41], affine=affine, header=hdr)
    return output


img_T1GD_left_cat, img_T1GD_right_cat = t1gd_up_down_cat(img_T1GD_left), t1gd_up_down_cat(img_T1GD_right)

img_TOF_left_cat, img_TOF_right_cat = tof_up_down_cat(img_TOF_left), tof_up_down_cat(img_TOF_right)
nib.save(img_T1GD_left_cat, save_T1GD_left)
nib.save(img_T1GD_right_cat, save_T1GD_right)
nib.save(img_TOF_left_cat, save_TOF_left)
nib.save(img_TOF_right_cat, save_TOF_right)
print('finish')
