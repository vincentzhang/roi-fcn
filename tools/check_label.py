from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import _init_paths
from datasets.factory import get_imdb


IMDB="ins_train_partial"
CLASSES = ('background',
           'foreground')

def plot():
    # first read an image from the DB
    imdb = get_imdb(IMDB)
    #plt.ion() # turn on interactive mode, plt.show is non-blocking 
    #fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')
    #import pdb;pdb.set_trace()
    for i in xrange(imdb.num_images):
        # Load the demo image
        img = imdb.get_image(i)
        label = imdb.get_label(i)
        #bbox = imdb.get_bbox(i)
        #boxes = bbox['boxes'] 
        _ = raw_input("Press [enter] to continue.") # wait for input from the user

plot()
