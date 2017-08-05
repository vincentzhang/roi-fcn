import matplotlib
matplotlib.use('Agg')

import numpy as np
import os

from matplotlib import pyplot as plt
import PIL
import cv2
import h5py

data_path = 'data/socket'
img_h5 = h5py.File(os.path.join(data_path, 'seg_band.h5'))
label_h5 = h5py.File(os.path.join(data_path, 'seg_band_mask.h5'))

if __name__ == '__main__':
    out_path = os.path.join(data_path, 'gt_overlay')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    vol_names = img_h5.keys()
    num_vols = len(vol_names)
    for i in xrange(num_vols):
        # for each volume, traverse all images
        vol = img_h5[vol_names[i]][:]
        for sliceidx in xrange(vol.shape[-1]):
            # Load the original image
            im = vol[:,:,sliceidx]
            im = np.dstack((im, im, im))
            # this label is binary
            label = label_h5[vol_names[i]+'_mask'][:,:,sliceidx]
            label = label * 255
            color_mask_label = [0, 97, 255] # blue
            label_img = np.ones( (label.shape[0], label.shape[1], 3)).astype('uint8')
            for j in range(3):
                label_img[:, :, j] = color_mask_label[j]
            label_img = np.dstack( (label_img, label*0.5) ).astype('uint8')
            # Visualization
            plt.imshow(im, interpolation='none')
            ax = plt.gca()
            # overlay
            ax.imshow(label_img, interpolation='none')
            plt.savefig(os.path.join(out_path, vol_names[i]+'_'+str(sliceidx)), bbox_inches='tight')
