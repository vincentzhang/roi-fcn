## Overview
This repo contains the code for ROI convolution based FCN. It is largely based on the [Faster R-CNN code](https://github.com/rbgirshick/py-faster-rcnn)

## Citing this work 

If you find Faster-RCNN-FCN useful in your research, please consider citing:

    @inproceedings{zhang2018end,
        title={End-to-end detection-segmentation network with ROI convolution},
        author={Zhang, Zichen and Tang, Min and Cobzas, Dana and Zonoobi, Dornoosh and Jagersand, Martin and Jaremko, Jacob L},
        booktitle={2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)},
        pages={1509--1512},
        year={2018},
        organization={IEEE}
    }


## Contents
1. [Installation](#installation)
3. [Demo](#demo)
5. [Training and testing](#training-and-testing-models)
6. [Usage](#usage)

### Basic installation 

1. Clone this repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/vincentzhang/roi-fcn.git 
  ```
  If you didn't clone with the `--recursive` flag, then you'll need to manually clone the `caffe-roi` submodule:
  ```Shell
    git submodule update --init --recursive
  ```

2. Build Caffe and pycaffe
  **Note:** Caffe *must* be built with support for Python layers!
    ```Shell
    # ROOT refers to the directory that you cloned this repo into.
    cd $ROOT/caffe-roi
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # In your Makefile.config, make sure to have this line uncommented
    WITH_PYTHON_LAYER := 1
    # Unrelatedly, it's also recommended that you use CUDNN
    USE_CUDNN := 1
    # Compile
    make -j8 && make pycaffe
    ```
  You can download my [Makefile.config](https://drive.google.com/open?id=1NSeWp7INxGWUrSdCTCwol8NmV_0-Ar5k) for reference.

3. Build the Cython modules
    ```Shell
    cd $ROOT/lib
    make
    ```

5. Download pre-computed Faster R-CNN detectors
    ```Shell
    cd $FRCN_ROOT
    ./data/scripts/fetch_faster_rcnn_models.sh
    ```

    This will populate the `$FRCN_ROOT/data` folder with `faster_rcnn_models`. See `data/README.md` for details.
    These models were trained on VOC 2007 trainval.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] If you want to use COCO, please see some notes under `data/README.md`
7. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
