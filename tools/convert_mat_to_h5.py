# convert from mat to h5
import os
import scipy.io as sio
import h5py
import glob

# define the path for the mat files
#mat_dat_path = '/data/dataset/hip/abhi/3D SegmentationMasks'
mat_dat_path = '/data/dataset/hip/abhi/band_3d_masks/3D SegmentationMasks'

# Images
f_list =  glob.glob(os.path.join(mat_dat_path, '*_00??.mat'))
#h5_f = h5py.File(os.path.join(mat_dat_path, '..', 'seg.h5'), 'w')
h5_f = h5py.File(os.path.join(mat_dat_path, '..', 'seg_band.h5'), 'w')
for f in f_list:
    h5_f.create_dataset(f.split('/')[-1].split('.')[0], data=sio.loadmat(f)['array3D'])
h5_f.close()

# Labels
f_list =  glob.glob(os.path.join(mat_dat_path, '*_00??_mask.mat'))
#h5_f = h5py.File(os.path.join(mat_dat_path, '..', 'seg_mask.h5'), 'w')
h5_f = h5py.File(os.path.join(mat_dat_path, '..', 'seg_band_mask.h5'), 'w')
for f in f_list:
    h5_f.create_dataset(f.split('/')[-1].split('.')[0], data=sio.loadmat(f)['mask3D'])
h5_f.close()

