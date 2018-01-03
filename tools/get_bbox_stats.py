from __future__ import print_function
import pickle
from PIL import Image
import numpy as np
import _init_paths
from datasets.factory import get_imdb
import pdb

IMDB="ins_train_partial"
CLASSES = ('background',
           'foreground')

def main():
    # first read an image from the DB
    imdb = get_imdb(IMDB)
    xmin = 1000
    xmax = 0
    ymin = 1000
    ymax = 0
    areas = []
    widths = []
    heights = []
    for i in xrange(imdb.num_images):
        # Load the demo image
        bbox = imdb.get_bbox(i)
        boxes = bbox['boxes']
        area = bbox['seg_areas']
        areas.extend(area)
        for box in boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            widths.append(w)
            heights.append(h)
            if w == 0 or h == 0:
                print("width or height is 0")
                #pdb.set_trace()
            xmin = min(box[0], xmin)
            ymin = min(box[1], ymin)
            xmax = max(box[2], xmax)
            ymax = max(box[3], ymax)
        #pdb.set_trace()
        #_ = raw_input("Press [enter] to continue.") # wait for input from the user
    # get stats
    mean_width = np.mean(widths)
    mean_height = np.mean(heights)
    mean_area = np.mean(areas)
    max_width = np.max(widths)
    min_width = np.min(widths)
    max_height = np.max(heights)
    min_height = np.min(heights)
    max_area = np.max(areas)
    min_area = np.min(areas)
    print("BBox statistics:")
    print("Width min,max,mean: {},{},{}".format(min_width, max_width, mean_width))
    print("Height min,max,mean: {},{},{}".format(min_height, max_height, mean_height))
    print("Area min,max,mean: {},{},{}".format(min_area, max_area, mean_area))
    print("x min,max: {},{}".format(xmin,xmax))
    print("y min,max: {},{}".format(ymin,ymax))


main()
