""" 
Python implementation of the color map function for the PASCAL VOC data set. 
Official Matlab version can be found in the PASCAL VOC devkit 
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def color_map_viz(ax=None):
    labels = ['bg', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    nclasses = 21
    row_size = 20
    col_size = 200
    cmap = color_map()
    # import pdb; pdb.set_trace()
    # cmap: [256, 3]
    array = np.empty((row_size, col_size*(nclasses+1), cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[:, i*col_size:i*col_size+col_size] = cmap[i]
    # for the class that is marked as "difficult"
    array[:, nclasses*col_size:nclasses*col_size+col_size] = cmap[-1]
    
    if ax is not None:
        # given axis
        ax.imshow(array)
        ax.set_xticklabels(labels)
        ax.set_xticks( [col_size*i+col_size/2 for i in range(nclasses+1)] )
        ax.set_yticks([])
    else:
        imshow(array)
        plt.xticks([col_size*i+col_size/2 for i in range(nclasses+1)], labels)
        plt.yticks([])
        plt.show()

if __name__ == "__main__":
    color_map_viz()
