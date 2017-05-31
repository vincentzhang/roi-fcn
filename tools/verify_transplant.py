import caffe
import surgery

import numpy as np
import os
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

base_weights = '../data/imagenet_models/VGG16.v2.fcn-surgery.caffemodel'
weights = '../data/imagenet_models/VGG16.v2.fcn-surgery-all.caffemodel'
#cfg_file= "../experiments/cfgs/faster_rcnn_end2end.yml"
cfg_file= "../experiments/cfgs/detect_end2end.yml"
cfg_from_file(cfg_file)
base_net = caffe.Net('../models/pascal_voc/VGG16/detect_end2end/train.prototxt',
        base_weights,
        caffe.TEST)


#cfg_file= "../experiments/cfgs/detect_end2end.yml"
#cfg_from_file(cfg_file)
net = caffe.Net('../models/pascal_voc/VGG16/detect_end2end/train.prototxt',
        weights,
        caffe.TEST)


# set up caffe
#caffe.set_mode_gpu()
#caffe.set_device(0)

for layer in base_net.params.keys():
    print("Net 0  :  histogram of layer {}: {} ".format(layer,
        np.histogram(base_net.params[layer][0].data, [-1, -0.5, 0, 0.2, 0.5, 0.8,
            1.0, 1000])[0]))
    print("Net 1  :  histogram of layer {}: {} ".format(layer,
        np.histogram(net.params[layer][0].data, [-1, -0.5, 0, 0.2, 0.5, 0.8,
            1.0, 1000])[0]))
        # verify that VGG-surgery 1 is the same as 2. the diff: with or w/o the
        # yml file of detect_end2end.yml
