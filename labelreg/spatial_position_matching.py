import nibabel as nib
import os


data_path_ref_image = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\full\17'
data_path_aim_image = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\full\18'
filename_T1GD_ref_image = os.path.join(data_path_ref_image, 'T1GD.nii.gz')
filename_T1GD_aim_image = os.path.join(data_path_aim_image, 'T1GD.nii.gz')
filename_TOF_ref_image = os.path.join(data_path_ref_image, 'TOF.nii.gz')
filename_TOF_aim_image = os.path.join(data_path_aim_image, 'TOF.nii.gz')

save_path_T1GD_aim = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\full\18\T1GD.nii.gz'
save_path_TOF_aim = r'C:\Users\Administrator\Desktop\reg_tutorials\code_data\data\jdm\full\18\TOF.nii.gz'


img_T1GD_ref = nib.load(filename_T1GD_ref_image)
img_T1GD_aim = nib.load(filename_T1GD_aim_image)
img_TOF_ref = nib.load(filename_TOF_ref_image)
img_TOF_aim = nib.load(filename_TOF_aim_image)
img_T1GD_ref_get_data = img_T1GD_ref.get_data()
img_T1GD_aim_get_data = img_T1GD_aim.get_data()
img_TOF_ref_get_data = img_TOF_ref.get_data()
img_TOF_aim_get_data = img_TOF_aim.get_data()
affine_T1GD_ref = img_T1GD_ref.affine.copy()
affine_TOF_ref = img_TOF_ref.affine.copy()
hdr_T1GD = img_T1GD_aim.header.copy()
hdr_TOF = img_TOF_aim.header.copy()


nib.save(nib.Nifti1Image(img_T1GD_aim_get_data, affine_T1GD_ref, header=hdr_T1GD), save_path_T1GD_aim)
nib.save(nib.Nifti1Image(img_TOF_aim_get_data, affine_TOF_ref, header=hdr_TOF), save_path_TOF_aim)
print('finish')


