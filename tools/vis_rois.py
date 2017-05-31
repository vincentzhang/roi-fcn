from __future__ import print_function
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import sys
sys.path.append('./lib/utils/')
from blob import prep_im_for_blob
img = '/data/repo/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/004622.jpg'


# Create figure and axes
#fig,ax = plt.subplots(1)

def resize(im):
    im, im_scale = prep_im_for_blob(im, np.array([[[0, 0, 0]]]), 600,1000)
    im = im/float(255)
    return im, im_scale

def plot():
    # draw rectangle of the image region
    im = np.array(Image.open(img), dtype=np.uint8)
    import pdb
    #pdb.set_trace()
    im, _ = resize(im)
    with open('rois.p', 'rb') as f:
        rois = pickle.load(f)

    #rois = np.array([[ 38,  60, 374, 499],
    #           [ 44,  99, 364, 499]], dtype=np.uint16)
    width = 375
    height = 500
    alt.ion() # turn on interactive mode, plt.show is non-blocking 
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    # Display the image
    ax.imshow(im)
    # draw rois
    i = 0
    #for roi in rois:
    #    roi_rect = Rectangle( (roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1],
    #           facecolor='red', alpha=0.1,  edgecolor='red', linewidth=1, label='class = {}'.format(roi[0]))
    #    i += 1
    #    ax.add_patch(roi_rect)
    #ax.legend(loc=2, ncol=1, bbox_to_anchor=(1.05, 1))
    #plt.savefig("vis_rois_resize.png", bbox_inches="tight")
    for roi in rois:
        if roi[0] == 0: #background class
            #continue
            roi_rect = Rectangle( (roi[2], roi[3]), roi[4]-roi[2], roi[5]-roi[3],
               alpha=0.1, facecolor="blue", edgecolor="blue" )
        else:
            roi_rect = Rectangle( (roi[2], roi[3]), roi[4]-roi[2], roi[5]-roi[3],
               facecolor='red', alpha=0.1,edgecolor="red" , linewidth=1, label='class = {}'.format(roi[0]))
               #facecolor="none", edgecolor="Red", linewidth=3 )
            i += 1
        ax.add_patch(roi_rect)
        plt.show()
        _ = raw_input("Press [enter] to continue.") # wait for input from the user
        #plt.close()    # close the figure to show the next one.
    #plt.savefig("roi_resize.png", bbox_inches="tight")



plot()
