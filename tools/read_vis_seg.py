import skimage.io
import matplotlib.pyplot as plt
import pdb

def main(img_name):
    img = skimage.io.imread(img_name)
    #pdb.set_trace()
    plt.imshow(img)
    plt.show()

img = '/data/dataset/VOCdevkit/VOC2007/SegmentationClass/000032.png'
main(img)
