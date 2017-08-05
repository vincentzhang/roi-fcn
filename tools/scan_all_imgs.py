""" This script scans the pred socket directory and check whether it contains and positive pixels """
from PIL import Image
import numpy as np
import os

import pdb

#img_path = "data/socket/pred_socket"
img_path = "data/socket/pred_socket4000"
def main():
    total = 0
    for root, _, files in os.walk(img_path):
        for f in files:
            if '.png' in f:
                img = np.asarray(Image.open(os.path.join(root, f)))
                total += np.count_nonzero(img)
    print("The total positives are: {}".format(total))

main()
