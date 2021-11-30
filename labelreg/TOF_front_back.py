import nibabel as nib
import os

data_path_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\only_left\31'
data_path_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\only_right\31'
data_path_T1GD = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\T1GD_left_right_up_down\31'
filename_T1GD_left = os.path.join(data_path_T1GD, 'T1GD_left_ud.nii.gz')
filename_TOF_left = os.path.join(data_path_left, 'TOF_left.nii.gz')
filename_T1GD_right = os.path.join(data_path_T1GD, 'T1GD_right_ud.nii.gz')
filename_TOF_right = os.path.join(data_path_right, 'TOF_right.nii.gz')


save_path_TOF_left_fb = r"C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\TOF_left_right_front_back\31\TOF_left_fb.nii.gz"
save_path_TOF_right_fb = r"C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\TOF_left_right_front_back\31\TOF_right_fb.nii.gz"


img_T1GD_left = nib.load(filename_T1GD_left)
img_TOF_left = nib.load(filename_TOF_left)
img_T1GD_right = nib.load(filename_T1GD_right)
img_TOF_right = nib.load(filename_TOF_right)
img_T1GD_left_get_data = img_T1GD_left.get_data()
img_T1GD_right_get_data = img_T1GD_right.get_data()
img_TOF_left_get_data = img_TOF_left.get_data()
img_TOF_right_get_data = img_TOF_right.get_data()
hdr_TOF_left = img_TOF_left.header.copy()
hdr_TOF_right = img_TOF_right.header.copy()
affine_left_T1GD = img_T1GD_left.affine.copy()
affine_right_T1GD = img_T1GD_right.affine.copy()
affine_left_TOF = img_TOF_left.affine.copy()
affine_right_TOF = img_TOF_right.affine.copy()

left_y_word_coordinate = 1*affine_left_T1GD[1][2] + affine_left_T1GD[1][3]
TOF_left_y_img_coordinate = int((left_y_word_coordinate - affine_left_TOF[1][3]) // affine_left_TOF[1][1])
right_y_word_coordinate = 1*affine_right_T1GD[1][2] + affine_right_T1GD[1][3]
TOF_right_y_img_coordinate = int((right_y_word_coordinate - affine_right_TOF[1][3]) // affine_right_TOF[1][1])

correct = -73
affine_left_TOF[1][3] = affine_left_TOF[1][2]-affine_left_TOF[1][1]*(TOF_left_y_img_coordinate+correct)
affine_right_TOF[1][3] = affine_right_T1GD[1][2]-affine_right_T1GD[1][2]*(TOF_right_y_img_coordinate+correct)

def TOF_front_back_cat(TOF_left_data, TOF_right_data, TOF_left_coordinate, TOF_right_coordinate, T1GD_data):
    img_TOF_left_front_back = nib.Nifti1Image(TOF_left_data[:, TOF_left_coordinate:TOF_left_coordinate+\
                                            T1GD_data.shape[2], :], affine=affine_left_TOF, header=hdr_TOF_left)
    img_TOF_right_front_back = nib.Nifti1Image(TOF_right_data[:, TOF_right_coordinate:TOF_right_coordinate+\
                                            T1GD_data.shape[2], :], affine=affine_right_TOF, header=hdr_TOF_right)

    return img_TOF_left_front_back, img_TOF_right_front_back


TOlfb, TFrfb = TOF_front_back_cat(img_TOF_left_get_data, img_TOF_right_get_data, TOF_left_y_img_coordinate, \
                                TOF_right_y_img_coordinate, img_T1GD_left_get_data)
nib.save(TOlfb, save_path_TOF_left_fb)
nib.save(TFrfb, save_path_TOF_right_fb)

print('finish')
