import nibabel as nib
import os


data_path_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\only_left\31'
data_path_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\only_right\31'
filename_T1GD_left = os.path.join(data_path_left, 'T1GD_left.nii.gz')
filename_TOF_left = os.path.join(data_path_left, 'TOF_left.nii.gz')
filename_T1GD_right = os.path.join(data_path_right, 'T1GD_right.nii.gz')
filename_TOF_right = os.path.join(data_path_right, 'TOF_right.nii.gz')


save_path_T1GD_left_ud = r"C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\T1GD_left_right_up_down\31\T1GD_left_ud.nii.gz"
save_path_T1GD_right_ud = r"C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\T1GD_left_right_up_down\31\T1GD_right_ud.nii.gz"


img_T1GD_left = nib.load(filename_T1GD_left)
img_TOF_left = nib.load(filename_TOF_left)
img_T1GD_right = nib.load(filename_T1GD_right)
img_TOF_right = nib.load(filename_TOF_right)
img_T1GD_left_get_data = img_T1GD_left.get_data()
img_T1GD_right_get_data = img_T1GD_right.get_data()
img_TOF_left_get_data = img_TOF_left.get_data()
img_TOF_right_get_data = img_TOF_right.get_data()
hdr_T1GD_left = img_T1GD_left.header.copy()
hdr_T1GD_right = img_T1GD_right.header.copy()
affine_left_T1GD = img_T1GD_left.affine.copy()
affine_right_T1GD = img_T1GD_right.affine.copy()
affine_left_TOF = img_TOF_left.affine.copy()
affine_right_TOF = img_TOF_right.affine.copy()


left_z_word_coordinate = (img_TOF_left_get_data.shape[2] - 1) * affine_left_TOF[2][2] + affine_left_TOF[2][3]
T1GD_left_y_img_coordinate = int((left_z_word_coordinate - affine_left_T1GD[2][3]) // affine_left_T1GD[2][1])
right_z_word_coordinate = (img_TOF_right_get_data.shape[2] - 1) * affine_right_TOF[2][2] + affine_right_TOF[2][3]
T1GD_right_y_img_coordinate = int((right_z_word_coordinate - affine_right_T1GD[2][3]) // affine_right_T1GD[2][1])


correct = 0
affine_left_T1GD[2][3] = affine_left_T1GD[2][3]+affine_left_T1GD[2][1]*(T1GD_left_y_img_coordinate+correct)
affine_right_T1GD[2][3] = affine_right_T1GD[2][3]+affine_left_T1GD[2][1]*(T1GD_right_y_img_coordinate+correct)

def T1GD_up_down_cat(T1GD_left_data, T1GD_right_data, T1GD_left_coordinate, T1GD_right_coordinate, TOF_data):

    img_T1GD_left_up_down = nib.Nifti1Image(T1GD_left_data[:, T1GD_left_coordinate:T1GD_left_coordinate +\
                                    TOF_data.shape[2], :], affine=affine_left_T1GD, header=hdr_T1GD_left)
    img_T1GD_right_up_down = nib.Nifti1Image(T1GD_right_data[0:, T1GD_right_coordinate:T1GD_right_coordinate\
                                    +TOF_data.shape[2], :], affine=affine_right_T1GD, header=hdr_T1GD_right)

    return img_T1GD_left_up_down, img_T1GD_right_up_down


TDlup, TDrup = T1GD_up_down_cat(img_T1GD_left_get_data, img_T1GD_right_get_data, T1GD_left_y_img_coordinate, \
                                T1GD_right_y_img_coordinate, img_TOF_left_get_data)


nib.save(TDlup, save_path_T1GD_left_ud)
nib.save(TDrup, save_path_T1GD_right_ud)

print('finish')
