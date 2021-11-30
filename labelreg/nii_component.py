import nibabel as nib
import os


data_path= r'C:\Users\Administrator\Desktop'
save_path = r'C:\Users\Administrator\Desktop\component.nii.gz'
img = os.path.join(data_path, 'case000000.nii.gz')
img = nib.load(img)
img1 = img.slicer[:, :, :, 0]
nii_data = img1.get_data()
new_data = nii_data.copy()


new_data[new_data>0] = 1


affine = img1.affine.copy()
hdr = img1.header.copy()


nib.save(nib.Nifti1Image(new_data, affine, hdr), save_path)
print('finish')