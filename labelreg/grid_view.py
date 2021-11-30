
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib

# from skimage.transform import resize

data_path = r'C:\Users\Administrator\Desktop\save\test1'
data_file = os.path.join(data_path, 'ddf3.nii.gz')
img_load_ddf = nib.load(data_file)
ddf = img_load_ddf.get_data()
a = ddf.shape[0]
b = ddf.shape[1]
c = ddf.shape[2]
ref_grid = np.stack(np.meshgrid([i for i in range(a)],
                                [j for j in range(b)],
                                [k for k in range(c)],
                                indexing='ij'), axis=3)
# a = ref_grid[...,0]
# b = ref_grid[...,1]
# c = ref_grid[...,2]
grid_warped = ref_grid + ddf

#
grid_warped_x = grid_warped[..., 0]
grid_warped_y = grid_warped[..., 1]
grid_warped_z = grid_warped[..., 2]


def grid2contour(grid_x, grid_z):
    '''
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    '''
    # assert grid.ndim == 3
    X, Y = np.meshgrid([i for i in range(64)], [j for j in range(112)])

    Z1 = grid_x[:, :, 34]
    # Z1 = Z1[::-1]
    Z2 = grid_z[:, 34, :]
    # Z2 = Z2[::-1]
    # Z1 = grid[:, :, 0] + 2  # remove the dashed line
    # Z1 = Z1[::-1]  # vertical flip
    # Z2 = grid[:, :, 1] + 2

    plt.figure()

    # plt.contourf(X, Y, Z1)
    plt.contour(X, Y, Z2, 20, colors='k')
    # plt.clabel(CS, fontsize=9, inline=1)

    # plt.contourf(X, Y, Z2)
    # plt.contour(X, Y, Z2, 15, colors='k')
    # plt.clabel(CS, fontsize=9, inline=1)
    plt.contour(X, Y, Z1, 20, colors='k')
    plt.xticks(()), plt.yticks(())  # remove x, y ticks
    plt.title('DVF')
    plt.show()


grid2contour(grid_warped_x, grid_warped_z)

