
import nibabel as nib
import os


data_path = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\full\31'
filename_T1GD = os.path.join(data_path, 'T1GD.nii.gz')
filename_TOF = os.path.join(data_path, 'TOF.nii.gz')


save_path_T1GD_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\only_left\31\T1GD_left.nii.gz'
save_path_T1GD_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\only_right\31\T1GD_right.nii.gz'
save_path_TOF_left = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\only_left\31\TOF_left.nii.gz'
save_path_TOF_right = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\only_right\31\TOF_right.nii.gz'


img_T1GD = nib.load(filename_T1GD)
img_TOF = nib.load(filename_TOF)
img_T1GD_get_data = img_T1GD.get_data()
img_TOF_get_data = img_TOF.get_data()
hdr_T1GD = img_T1GD.header.copy()
hdr_TOF = img_TOF.header.copy()
affine_left_T1GD = img_T1GD.affine.copy()
affine_left_TOF = img_TOF.affine.copy()

def image_left_cat(T1GD_data, TOF_data):
    img_T1GD_left = nib.Nifti1Image(T1GD_data[0:T1GD_data.shape[0]//2, :, :], affine=affine_left_T1GD, header=hdr_T1GD)
    img_TOF_left = nib.Nifti1Image(TOF_data[0:TOF_data.shape[0]//2, :, :], affine=affine_left_TOF, header=hdr_TOF)

    return img_T1GD_left, img_TOF_left




correct = -5
affine_left_TOF[0][3] = affine_left_TOF[0][3]+correct*affine_left_TOF[0][0]
TDl, TFl = image_left_cat(img_T1GD_get_data, img_TOF_get_data)
nib.save(TDl, save_path_T1GD_left)
nib.save(TFl, save_path_TOF_left)


affine_left_T1GD[0][3] = affine_left_T1GD[0][3]+img_TOF_get_data.shape[0]//2*affine_left_T1GD[0][0]
affine_left_TOF[0][3] = affine_left_TOF[0][3]+img_TOF_get_data.shape[0]//2*affine_left_TOF[0][0]
affine_right_T1GD = affine_left_T1GD
affine_right_TOF = affine_left_TOF




def image_right_cat(T1GD_data, TOF_data):
    img_T1GD_right = nib.Nifti1Image(T1GD_data[T1GD_data.shape[0] // 2:T1GD_data.shape[1]+1, :, :],\
                                     affine=affine_right_T1GD, header=hdr_T1GD)
    img_TOF_right = nib.Nifti1Image(TOF_data[TOF_data.shape[0] // 2:TOF_data.shape[0]+1, :, :], \
                                    affine=affine_right_TOF, header=hdr_TOF)

    return img_T1GD_right, img_TOF_right


TDr, TFr = image_right_cat(img_T1GD_get_data,img_TOF_get_data)
nib.save(TDr, save_path_T1GD_right)
nib.save(TFr, save_path_TOF_right)
print('over')


