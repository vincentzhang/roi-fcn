import caffe
import surgery

import numpy as np
import os
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

weights = '../data/imagenet_models/VGG16.v2.caffemodel'
cfg_file= "../experiments/cfgs/faster_rcnn_end2end.yml"
cfg_from_file(cfg_file)
base_net = caffe.Net('../models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt',
        weights,
        caffe.TEST)

# set up caffe
caffe.set_mode_gpu()
caffe.set_device(0)

cfg_file= "../experiments/cfgs/detect_end2end.yml"
cfg_from_file(cfg_file)
net = caffe.Net('../models/pascal_voc/VGG16/detect_end2end/train.prototxt',
        caffe.TEST)
surgery.transplant(net, base_net)

# surgeries
interp_layers = [k for k in net.params.keys() if 'up' in k]
surgery.interp(net, interp_layers)

# Save the model
net.save('../data/imagenet_models/VGG16.v2.fcn-surgery-all.caffemodel')

# verify
for layer in net.params.keys():
    if layer in base_net.params.keys():
        print("Net 0  :  histogram of layer {}: {} ".format(layer,
            np.histogram(base_net.params[layer][0].data, [-1, -0.5, 0, 0.2, 0.5, 0.8,
                1.0, 1000])[0]))
        print("Net 1  :  histogram of layer {}: {} ".format(layer,
            np.histogram(net.params[layer][0].data, [-1, -0.5, 0, 0.2, 0.5, 0.8,
                1.0, 1000])[0]))
    else:
        print("Only Net 1  :  histogram of layer {}: {} ".format(layer,
            np.histogram(net.params[layer][0].data, [-1, -0.5, 0, 0.2, 0.5, 0.8,
                1.0, 1000])[0]))
