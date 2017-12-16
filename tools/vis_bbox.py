from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
#import sys
#sys.path.append('../lib/utils/')
import _init_paths
#from blob import prep_im_for_blob
from datasets.factory import get_imdb


# image source
#img = '/data/repo/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/004622.jpg'

IMDB="ins_train_partial"
CLASSES = ('background',
           'foreground')

# Create figure and axes
#fig,ax = plt.subplots(1)

def resize(im):
    im, im_scale = prep_im_for_blob(im, np.array([[[0, 0, 0]]]), 600,1000)
    im = im/float(255)
    return im, im_scale

def plot():
    # first read an image from the DB
    imdb = get_imdb(IMDB)
    plt.ion() # turn on interactive mode, plt.show is non-blocking 
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    #img = imdb.get_image(0)
    # matplotlib.image.AxesImage object 
    #imgobj = ax.imshow(img[:,:,::-1]) # BGR to RGB
    #import pdb;pdb.set_trace()
    for i in xrange(imdb.num_images):
        # Load the demo image
        img = imdb.get_image(i)
        label = imdb.get_label(i)
        bbox = imdb.get_bbox(i)
        boxes = bbox['boxes'] 
        num_boxes = boxes.shape[0]
        # draw rectangle of the image region
        #boxes = np.array([[ 38,  60, 374, 499],
        #           [ 44,  99, 364, 499]], dtype=np.uint16)
        # Display the image
        #imgobj.set_data(img[:,:,::-1])
        ax.clear()
        ax.imshow(img[:,:,::-1]) # BGR to RGB
        ax.set_title('The {}-th image, {} boxes'.format(i, num_boxes))
        plt.show()
        # draw rois
        #for roi in rois:
        #    roi_rect = Rectangle( (roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1],
        #           facecolor='red', alpha=0.1,  edgecolor='red', linewidth=1, label='class = {}'.format(roi[0]))
        #    i += 1
        #    ax.add_patch(roi_rect)
        #ax.legend(loc=2, ncol=1, bbox_to_anchor=(1.05, 1))
        #plt.savefig("vis_rois_resize.png", bbox_inches="tight")
        for box in boxes:
            # box: x1,y1,x2,y2
            rect = Rectangle( (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                   alpha=0.2, facecolor="blue", edgecolor="blue")
            patch = ax.add_patch(rect)
        _ = raw_input("Press [enter] to continue.") # wait for input from the user
        #ax.legend(loc=2, ncol=1, bbox_to_anchor=(1.05, 1))
        #plt.savefig("roi_resize.png", bbox_inches="tight")

# plt.close()    # close the figure to show the next one.
plot()
