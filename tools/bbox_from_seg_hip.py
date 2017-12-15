# generated bbox ground truth from pixel-wise segmentation
# For Hip data
# it currently only generate one bbox
from __future__ import print_function
import numpy as np
import h5py
import os
import pdb

mask_path = 'data/hip'
#f = h5py.File(os.path.join(mask_path, "seg_mask.h5"), 'r')
#bbox_path = 'data/socket/seg_bbox'
f = h5py.File(os.path.join(mask_path, "hip3d0_label.h5"), 'r')
bbox_path = 'data/hip/bbox'
if not os.path.exists(bbox_path):
    os.mkdir(bbox_path)

# dim:  shape (256, 391, 192), slices, height, width
for k in f.keys():
    data = np.array(f[k]) # convert to numpy
    with open( os.path.join(bbox_path, 'bbox_'+k+'.txt'), 'w') as bbox_file:
        # iterate through each slice
        for idx in range(data.shape[0]):
            mask = data[idx, :, :] # get the mask
            i,j = np.where(mask) # find positive mask
            if not i.size: # no positive mask
                print("{}_{},{}".format(k, idx, 0), file=bbox_file)
            else:
                h_min,w_min = np.min(zip(i,j), axis=0)
                h_max,w_max = np.max(zip(i,j), axis=0)
                print("{}_{},{},{},{},{},{}".format(k, idx, 1, w_min, h_min, w_max,
                    h_max), file=bbox_file)

