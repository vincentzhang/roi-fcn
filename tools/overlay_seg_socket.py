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
    imdb_name = 'socket_test_all'
    imdb = get_imdb(imdb_name)
    out_path = os.path.join(imdb.data_path, 'pred_overlay')
    seg_path = os.path.join(imdb.data_path, 'pred_socket50400')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    num_images = len(imdb.image_index)
    vol_name, sliceidx = imdb.image_path_at(0).rsplit('_',1)
    im = imdb.image_h5f[vol_name][:,:,int(sliceidx)]
    im = np.dstack((im, im, im))
    fig, ax = plt.subplots()
    img = ax.imshow(im, interpolation='none')
    fig.canvas.draw()

    for i in xrange(num_images):
        print("Processing the {}/{} image".format(i, num_images))
        if i < 2020:
            continue
        if i > 2050:
            break
        vol_name, sliceidx = imdb.image_path_at(i).rsplit('_',1)
        # Load the original image
        im = imdb.image_h5f[vol_name][:,:,int(sliceidx)]
        im = np.dstack((im, im, im))
        # this label is binary
        label = imdb.label_h5f[vol_name+'_mask'][:,:,int(sliceidx)]
        label = label * 255
        color_mask_label = [0, 97, 255] # blue
        label_img = np.ones( (label.shape[0], label.shape[1], 3)).astype('uint8')
        for j in range(3):
            label_img[:, :, j] = color_mask_label[j]
        label_img = np.dstack( (label_img, label*0.5) ).astype('uint8')
        # Load the prediction, in png format
        pred = np.asarray(PIL.Image.open(os.path.join(seg_path,
            imdb.image_index[i] + '.png')))
        pred = np.where(pred > 127, 255, 0)
        color_mask_pred = [255, 0, 114] # pink
        pred_img = np.ones( (pred.shape[0], pred.shape[1], 3) ).astype('uint8')
        for j in range(3):
            pred_img[:, :, j] = color_mask_pred[j]
        pred_img = np.dstack( (pred_img, pred*0.5) ).astype('uint8')
        # Visualization
        #plt.imshow(im, interpolation='none')
        #ax = plt.gca()
        img.set_data(im)
        # overlay
        img.set_data(label_img)
        img.set_data(pred_img)
        #ax.imshow(label_img, interpolation='none')
        #ax.imshow(pred_img, interpolation='none')
        #plt.show()
        #import pdb;pdb.set_trace()
        plt.savefig(os.path.join(out_path, imdb.image_index[i]), bbox_inches='tight')
        #fig.canvas.draw()
