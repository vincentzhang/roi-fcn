import matplotlib
matplotlib.use('Agg')

import _init_paths
from datasets.factory import get_imdb
import numpy as np
import os

from matplotlib import pyplot as plt
import PIL
import cv2

if __name__ == '__main__':
    imdb_name = 'hip_test'
    imdb = get_imdb(imdb_name)
    out_path = os.path.join(imdb.data_path, 'pred_overlay')
    seg_path = os.path.join(imdb.data_path, 'pred0')

    num_images = len(imdb.image_index)
    for i in xrange(num_images):
        # Load the original image
        im = cv2.imread(imdb.image_path_at(i))
        # Load the label, in jpeg format
        label = np.asarray(PIL.Image.open(imdb.label_path_at(i)))
        # binarize the label
        label = np.where(label > 127, 255, 0)
        #if label.max() == 0:
        #    continue
        color_mask_label = [0, 97, 255] # blue
        label_img = np.ones( (label.shape[0], label.shape[1], 3)).astype('uint8')
        for j in range(3):
            label_img[:, :, j] = color_mask_label[j]
        label_img = np.dstack( (label_img, label*0.2) ).astype('uint8')
        # Load the prediction, in png format
        pred = np.asarray(PIL.Image.open(os.path.join(seg_path,
            imdb.image_index[i] + '.png')))
        pred = np.where(pred > 127, 255, 0)
        color_mask_pred = [255, 0, 114] # pink
        pred_img = np.ones( (pred.shape[0], pred.shape[1], 3) ).astype('uint8')
        for j in range(3):
            pred_img[:, :, j] = color_mask_pred[j]
        pred_img = np.dstack( (pred_img, pred*0.2) ).astype('uint8')
        # Visualization
        plt.imshow(im, interpolation='none')
        ax = plt.gca()
        # overlay
        ax.imshow(label_img, interpolation='none')
        ax.imshow(pred_img, interpolation='none')
        #plt.show()
        #import pdb;pdb.set_trace()
        plt.savefig(os.path.join(out_path, imdb.image_index[i]), bbox_inches='tight')
        #break
