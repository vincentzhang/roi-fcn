import os
import numpy as np
import h5py
import pdb

# This is to get the mean value of the new dataset
fo = open('/data/dataset/hip/accetabulum/train.txt', 'r')
flist = []
for f in fo:
    flist.append(f.strip())
fo = open('/data/dataset/hip/accetabulum/test.txt', 'r')
for f in fo:
    flist.append(f.strip())

print("num of volumes {}".format(len(flist)))
fo.close()

h5f = h5py.File('../data/acce/resized_vols_2d.h5','r')
count= 0
avg = 0
for k in flist:
    img = h5f[k][...]
    count += 1
    print("Processing image {}".format(k))
    # have a running avg
    avg = avg + (img.mean()-avg)/count
print("The mean value is: {}".format(avg))
h5f.close()
