#!/usr/bin/env python
"""Demo of ROI-FCN on automatic acetabulum segmentation"""

import _init_paths
from autoseg.autoseg import autoseg, load_model
import cv2
import os, sys
import glob

if __name__ == '__main__':
    cwd = os.getcwd()
    demo_dir = os.path.dirname(os.path.realpath(__file__))
    if cwd == demo_dir:
        print("Please run the script from the $ROOT directory: $ python tools/demo.py. Exiting ...")
        sys.exit()

    outdir = "./output/demo"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read images
    net = load_model()
    for img_f in glob.glob("data/samples/*.jpg"):
        print("Processing image {}".format(img_f))
        pred = autoseg(img_f, net)
        print("Saveing prediction")
        # Multiply 255 for visualization purpose
        cv2.imwrite(os.path.join(outdir,'pred_'+img_f.strip().rsplit('/',1)[1]), pred*255)
