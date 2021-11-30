import nibabel as nib
import os


data_path_img = r'C:\Users\Administrator\Desktop\train\TOF_images'
data_path_label = r'C:\Users\Administrator\Desktop\train\TOF_labels'
# data_path_list = data_path.split('\\')

filename_img = os.path.join(data_path_img, 'case000023.nii.gz')
filename_label = os.path.join(data_path_label, 'case000023.nii.gz')

save_img = r'C:\Users\Administrator\Desktop\train\save\TOF_images\case000023.nii.gz'
save_label = r'C:\Users\Administrator\Desktop\train\save\TOF_labels\case000023.nii.gz'



img_TOF = nib.load(filename_img)
label_TOF = nib.load(filename_label)

img_TOF_get_data = img_TOF.get_data()
label_TOF_get_data = label_TOF.get_data()

hdr_TOF_img = img_TOF.header.copy()
hdr_TOF_label = label_TOF.header.copy()

affine_TOF_img = img_TOF.affine.copy()
affine_TOF_label = label_TOF.affine.copy()


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
    output = nib.Nifti1Image(data[0:96, 0:data.shape[1]-8, :], affine=affine, header=hdr)
    return output




img_TOF_left_cat, img_TOF_right_cat = tof_front_back_cat(img_TOF), tof_front_back_cat(label_TOF)

nib.save(img_TOF_left_cat, save_img)
nib.save(img_TOF_right_cat, save_label)
print('finish')
