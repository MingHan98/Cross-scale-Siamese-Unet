import nibabel as nib
import os

data_path= r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\up_down_slicer\31'
# data_path_list = data_path.split('\\')
filename_T1GD_left = os.path.join(data_path, 'T1GD_left.nii.gz')
filename_T1GD_right = os.path.join(data_path, 'T1GD_right.nii.gz')
filename_TOF_left = os.path.join(data_path, 'TOF_left.nii.gz')
filename_TOF_right = os.path.join(data_path, 'TOF_right.nii.gz')
save_T1GD_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\x_front_back_cat\31\T1GD_left.nii.gz'
save_T1GD_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\x_front_back_cat\31\T1GD_right.nii.gz'
save_TOF_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\x_front_back_cat\31\TOF_left.nii.gz'
save_TOF_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\x_front_back_cat\31\TOF_right.nii.gz'


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



def t1gd_front_back_cat(input_image):
    data = input_image.get_data()
    hdr = input_image.header.copy()
    affine = input_image.affine.copy()
    output = nib.Nifti1Image(data[:, :, 0:data.shape[2]-18], affine=affine, header=hdr)
    return output


def tof_front_back_cat(input_image):
    data = input_image.get_data()
    hdr = input_image.header.copy()
    affine = input_image.affine.copy()
    output = nib.Nifti1Image(data[:, 0:data.shape[1]-18, :], affine=affine, header=hdr)
    return output


img_T1GD_left_cat, img_T1GD_right_cat = t1gd_front_back_cat(img_T1GD_left), t1gd_front_back_cat(img_T1GD_right)

img_TOF_left_cat, img_TOF_right_cat = tof_front_back_cat(img_TOF_left), tof_front_back_cat(img_TOF_right)
nib.save(img_T1GD_left_cat, save_T1GD_left)
nib.save(img_T1GD_right_cat, save_T1GD_right)
nib.save(img_TOF_left_cat, save_TOF_left)
nib.save(img_TOF_right_cat, save_TOF_right)
print('finish')
