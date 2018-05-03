# generated bbox ground truth from pixel-wise segmentation
# it currently only generate one bbox
from __future__ import print_function
import numpy as np
import h5py
import os
import pdb

mask_path = '../data/acce'
f = h5py.File(os.path.join(mask_path, "resized_label_ac_2d.h5"), 'r')

bbox_path = '../data/acce/bbox'
if not os.path.exists(bbox_path):
    os.mkdir(bbox_path)

# dim:  shape (256, 367, 342), slices, height, width
count = 0
for k in f.keys():
    #pdb.set_trace()
    count += 1
    print("processing {}-th vol".format(count))
    data = f[k][...] # convert to numpy
    k = k.rsplit('_',1)[0] # strip the '_label' from the vol name
    with open( os.path.join(bbox_path, k)+'_bbox.txt', 'w') as bbox_file:
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
f.close()
