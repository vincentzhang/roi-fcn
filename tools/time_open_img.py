import time
import os
import scipy.io as sio
import h5py

# open mat file 10 times and time
mat_dat_path = '/data/dataset/hip/abhi/3D SegmentationMasks'
start = time.time()
for i in range(10):
    cont = sio.loadmat(os.path.join(mat_dat_path, 'Za6301IM_0025.mat'))
time_mat = time.time() - start
print("mat: ", time_mat)

# time for opening h5 files
h5_dat_path = '/data/dataset/hip/abhi'
start = time.time()
f = h5py.File(os.path.join(h5_dat_path, 'seg.h5'))
for i in range(10):
    cont = f[f.keys()[i]]
time_h5 = time.time() - start
print("h5: ", time_h5)

# time for opening jpeg files
import cv2
import glob
jpg_dat_path = '/data/dataset/hip/train_seg_img'
f_list = glob.glob(os.path.join(jpg_dat_path, '*.jpg'))
start = time.time()
for i in range(10):
    cont = cv2.imread(f_list[i])
time_jpg = time.time() - start
print("jpg: ", time_jpg)


