import os
import nibabel as nib
import numpy as np
import elasticdeform as etf

data_path = r'E:\data\MICCAI_BraTS2020_TrainingData\BraTS20_Training_034'
save_path = r"C:\Users\Administrator\Desktop\paper_pictures\brain7"

data_join = os.path.join(data_path,'BraTS20_Training_034_t2.nii.gz')

def t1gd_label_information(file_t1gd):
    T1GD_label = nib.load(file_t1gd)
    T1GD_label_data = T1GD_label.get_data()
    T1GD_labe_affine = T1GD_label.affine.copy()
    T1GD_labe_hdr = T1GD_label.header.copy()


    return T1GD_label_data, T1GD_labe_affine, T1GD_labe_hdr

T2_data, T2_affine, T2_hdr = t1gd_label_information(data_join)


displacement = np.random.randn(3, 3, 3, 3) * 1.0

# apply deformation with a random 3 x 3 grid
T2_deformed = etf.deform_random_grid(T2_data, displacement)

save_T1GD_labe0 = nib.Nifti1Image(T2_deformed, T2_affine, T2_hdr)
nib.save(save_T1GD_labe0, os.path.join(save_path, 'case000000_warp.nii.gz'))



print('over')
