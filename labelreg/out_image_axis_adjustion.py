import os
import numpy as np
import nibabel as nib

data_path = r"C:\Users\Administrator\Desktop\save\test1"
data_test_path = r"C:\Users\Administrator\Desktop\reg_tutorials\code_data\jdm_data\else\cross4\test\TOF_images"
save_path = r"C:\Users\Administrator\Desktop\save\test1"

test_image0 = os.path.join(data_test_path, 'case000002.nii.gz')
test_image1 = os.path.join(data_test_path, 'case000016.nii.gz')
test_image2 = os.path.join(data_test_path, 'case000018.nii.gz')
test_image3 = os.path.join(data_test_path, 'case000021.nii.gz')
test_image4 = os.path.join(data_test_path, 'case000022.nii.gz')
ddf0 = os.path.join(data_path, 'ddf0.nii.gz')
ddf1 = os.path.join(data_path, 'ddf1.nii.gz')
ddf2 = os.path.join(data_path, 'ddf2.nii.gz')
ddf3 = os.path.join(data_path, 'ddf3.nii.gz')
ddf4 = os.path.join(data_path, 'ddf4.nii.gz')
warped_image0 = os.path.join(data_path, 'warped_image0.nii.gz')
warped_image1 = os.path.join(data_path, 'warped_image1.nii.gz')
warped_image2 = os.path.join(data_path, 'warped_image2.nii.gz')
warped_image3 = os.path.join(data_path, 'warped_image3.nii.gz')
warped_image4 = os.path.join(data_path, 'warped_image4.nii.gz')
warped_label0 = os.path.join(data_path, 'warped_label0.nii.gz')
warped_label1 = os.path.join(data_path, 'warped_label1.nii.gz')
warped_label2 = os.path.join(data_path, 'warped_label2.nii.gz')
warped_label3 = os.path.join(data_path, 'warped_label3.nii.gz')
warped_label4 = os.path.join(data_path, 'warped_label4.nii.gz')


img_test_image0 = nib.load(test_image0)
img_test_image1 = nib.load(test_image1)
img_test_image2 = nib.load(test_image2)
img_test_image3 = nib.load(test_image3)
img_test_image4 = nib.load(test_image4)
img_ddf0 = nib.load(ddf0)
img_ddf1 = nib.load(ddf1)
img_ddf2 = nib.load(ddf2)
img_ddf3 = nib.load(ddf3)
img_ddf4 = nib.load(ddf4)
img_warped_image0 = nib.load(warped_image0)
img_warped_image1 = nib.load(warped_image1)
img_warped_image2 = nib.load(warped_image2)
img_warped_image3 = nib.load(warped_image3)
img_warped_image4 = nib.load(warped_image4)
img_warped_label0 = nib.load(warped_label0)
img_warped_label1 = nib.load(warped_label1)
img_warped_label2 = nib.load(warped_label2)
img_warped_label3 = nib.load(warped_label3)
img_warped_label4 = nib.load(warped_label4)


img_test0_affine = img_test_image0.affine.copy()
img_test1_affine = img_test_image1.affine.copy()
img_test2_affine = img_test_image2.affine.copy()
img_test3_affine = img_test_image3.affine.copy()
img_test4_affine = img_test_image4.affine.copy()
img_test0_hdr = img_test_image0.header.copy()
img_test1_hdr = img_test_image1.header.copy()
img_test2_hdr = img_test_image2.header.copy()
img_test3_hdr = img_test_image3.header.copy()
img_test4_hdr = img_test_image4.header.copy()


def axis_adjust(input_, affine_default):
    hdr_image = input_.header.copy()
    image_data = input_.get_data()
    affine = affine_default
    output = nib.Nifti1Image(image_data, affine, hdr_image)
    return output

# data_ddf0 = np.flip(np.transpose(img_ddf0.dataobj, (0, 2, 1, 4)), 2)
# data_ddf1 = np.flip(np.transpose(img_ddf1.dataobj, (0, 2, 1, 4)), 2)
# data_ddf0 = img_ddf0.dataobj
# data_ddf1 = img_ddf1.dataobj
# data_ddf2 = img_ddf2.dataobj
# data_ddf3 = img_ddf3.dataobj
# data_ddf4 = img_ddf4.dataobj

data_warped_image0 = np.flip(np.transpose(np.squeeze(img_warped_image0.dataobj), (0, 2, 1)), 2)
data_warped_image1 = np.flip(np.transpose(np.squeeze(img_warped_image1.dataobj), (0, 2, 1)), 2)
data_warped_image2 = np.flip(np.transpose(np.squeeze(img_warped_image2.dataobj), (0, 2, 1)), 2)
data_warped_image3 = np.flip(np.transpose(np.squeeze(img_warped_image3.dataobj), (0, 2, 1)), 2)
data_warped_image4 = np.flip(np.transpose(np.squeeze(img_warped_image4.dataobj), (0, 2, 1)), 2)
data_warped_label0 = np.flip(np.transpose(np.squeeze(img_warped_label0.dataobj), (0, 2, 1)), 2)
data_warped_label1 = np.flip(np.transpose(np.squeeze(img_warped_label1.dataobj), (0, 2, 1)), 2)
data_warped_label2 = np.flip(np.transpose(np.squeeze(img_warped_label2.dataobj), (0, 2, 1)), 2)
data_warped_label3 = np.flip(np.transpose(np.squeeze(img_warped_label3.dataobj), (0, 2, 1)), 2)
data_warped_label4 = np.flip(np.transpose(np.squeeze(img_warped_label4.dataobj), (0, 2, 1)), 2)


ddf0, ddf1, ddf2, ddf3, ddf4 = axis_adjust(img_ddf0, img_test0_affine), axis_adjust(img_ddf1, img_test1_affine),\
                               axis_adjust(img_ddf2, img_test2_affine), axis_adjust(img_ddf3, img_test3_affine), \
                               axis_adjust(img_ddf4, img_test4_affine),
save_warped_image0 = nib.Nifti1Image(data_warped_image0, img_test0_affine, img_test0_hdr)
save_warped_label0 = nib.Nifti1Image(data_warped_label0, img_test0_affine, img_test0_hdr)
save_warped_image1 = nib.Nifti1Image(data_warped_image1, img_test1_affine, img_test1_hdr)
save_warped_label1 = nib.Nifti1Image(data_warped_label1, img_test1_affine, img_test1_hdr)
save_warped_image2 = nib.Nifti1Image(data_warped_image2, img_test2_affine, img_test2_hdr)
save_warped_label2 = nib.Nifti1Image(data_warped_label2, img_test2_affine, img_test2_hdr)
save_warped_image3 = nib.Nifti1Image(data_warped_image3, img_test3_affine, img_test3_hdr)
save_warped_label3 = nib.Nifti1Image(data_warped_label3, img_test3_affine, img_test3_hdr)
save_warped_image4 = nib.Nifti1Image(data_warped_image4, img_test4_affine, img_test4_hdr)
save_warped_label4 = nib.Nifti1Image(data_warped_label4, img_test4_affine, img_test4_hdr)
# data_warped_image0 = np.flip(np.transpose(np.squeeze(img_warped_image0.dataobj), (0, 2, 1), 2))
# data_warped_image1 = np.flip(np.transpose(np.squeeze(img_warped_image1.dataobj), (0, 2, 1)), 2)
# data_warped_label0 = np.flip(np.transpose(np.squeeze(img_warped_label0.dataobj), (0, 2, 1)), 2)
# data_warped_label1 = np.flip(np.transpose(np.squeeze(img_warped_label0.dataobj), (0, 2, 1)), 2)


nib.save(ddf0, os.path.join(save_path, 'ddf0.nii.gz'))
nib.save(ddf1, os.path.join(save_path, 'ddf1.nii.gz'))
nib.save(ddf2, os.path.join(save_path, 'ddf2.nii.gz'))
nib.save(ddf3, os.path.join(save_path, 'ddf3.nii.gz'))
nib.save(ddf4, os.path.join(save_path, 'ddf4.nii.gz'))
nib.save(save_warped_image0, os.path.join(save_path, 'warped_image0.nii.gz'))
nib.save(save_warped_label0, os.path.join(save_path, 'warped_label0.nii.gz'))
nib.save(save_warped_image1, os.path.join(save_path, 'warped_image1.nii.gz'))
nib.save(save_warped_label1, os.path.join(save_path, 'warped_label1.nii.gz'))
nib.save(save_warped_image2, os.path.join(save_path, 'warped_image2.nii.gz'))
nib.save(save_warped_label2, os.path.join(save_path, 'warped_label2.nii.gz'))
nib.save(save_warped_image3, os.path.join(save_path, 'warped_image3.nii.gz'))
nib.save(save_warped_label3, os.path.join(save_path, 'warped_label3.nii.gz'))
nib.save(save_warped_image4, os.path.join(save_path, 'warped_image4.nii.gz'))
nib.save(save_warped_label4, os.path.join(save_path, 'warped_label4.nii.gz'))
# nib.save(left_img_t1gd, os.path.join(save_path_img, 'T1GD_left_img.nii.gz'))
# nib.save(right_img_t1gd, os.path.join(save_path_img, 'T1GD_right_img.nii.gz'))
# nib.save(left_label_t1gd, os.path.join(save_path_label, 'T1GD_left_label.nii.gz'))
# nib.save(right_label_t1gd, os.path.join(save_path_label, 'T1GD_right_label.nii.gz'))
# nib.save(left_img_tof, os.path.join(save_path_img, 'TOF_left_img.nii.gz'))
# nib.save(right_img_tof, os.path.join(save_path_img, 'TOF_right_img.nii.gz'))
print('finish')