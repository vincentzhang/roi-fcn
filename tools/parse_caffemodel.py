import caffe
from caffe.proto import caffe_pb2
import numpy as np

srcmodel = "../data/imagenet_models/VGG16.v2.caffemodel"
f = open(srcmodel,'rb')
blob = caffe_pb2.NetParameter() #caffemodel is a NetParameter proto
blob.ParseFromString(f.read())
f.close()

for i in range(len(blob.layer)):
  print("Name  of layer {}: {}".format(i, blob.layer[i].name))  # first layer
  try:
    print("Histogram    : {}".format(np.histogram(blob.layer[i].blobs[0].data,
        [-1, -0.5, 0, 0.2, 0.5, 0.8, 1.0, 1000])[0]))
  except Exception as e:
    print("layer {}: {}".format(blob.layer[i].name, str(e)))
