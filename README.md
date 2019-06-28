## Overview
This repo contains the code for ROI-FCN: ROI convolution based FCN, described in the following paper.

If you find it useful in your research, please consider citing:

    @inproceedings{zhang2018end,
        title={End-to-end detection-segmentation network with ROI convolution},
        author={Zhang, Zichen and Tang, Min and Cobzas, Dana and Zonoobi, Dornoosh and Jagersand, Martin and Jaremko, Jacob L},
        booktitle={2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)},
        pages={1509--1512},
        year={2018},
        organization={IEEE}
    }

It is largely based on the [Faster R-CNN code](https://github.com/rbgirshick/py-faster-rcnn)
The key difference is that we add the [ROI convolution layer in caffe](https://github.com/vincentzhang/caffe-roi/tree/73bc351c318402635e7220211740b1d44170d13d)

Code is provided as-is, no updates expected.

## Contents
1. [Installation](#installation)
2. [Demo](#demo)
3. [Training and testing](#training-and-testing)

### Installation 

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

4. Download the ImageNet pre-trained VGG16 weights (adapted to be fully convolutional):
    ```Shell
    cd $ROOT/data/scripts
    ./fetch_vgg16_fcn.sh
    ```

    This will populate the `$ROOT/data/imagenet_models` folder with `VGG16.v2.fcn-surgery-all.caffemodel`.

### Demo

To run the demo, first download the pretrained weights:
```Shell
cd $ROOT/data/scripts
./fetch_socket_models.sh
```
Run the demo script:
```Shell
cd $ROOT
python ./tools/demo.py
```
The demo runs the segmentation network trained on the acetabulum data used in the paper.

To show the generalization of the algorithm, the input images stored in `$ROOT/data/samples` are anonymized clinical images that are not in the training or testing dataset.

### Training

We are not allowed to share the dataset due to privacy restrictions.
But we're providing the workflow for training on your own dataset and the key files that need to be modified:

1. Entry point: a bash script in the experiments directory that specifies some hyperparameters

    Example:
    ```Shell
    $ ./experiments/scripts/socket_scratch_n_1e-4_fg150_roils_end2end.sh 0 VGG16 socket
    ```

2. Most of the configs files that specifies the caffe solver and network would not be very different but
you would need to write you own data loader following this file as an example: `lib/datasets/socket.py`.
The function `gt_roidb()` generates or load a numpy file of the ground truth bounding boxes which you would need to create offline beforehand.

3. Create symlinks for your dataset

	```Shell
    cd $ROOT/data
    ln -s SOURCE_PATH_TO_YOUR_DATA TARGET_PATH
    ```

### Testing

The following code runs the trained models on the entire test dataset:

```Shell
./experiments/scripts/test_socket_scratch_n_1e-4_fg150_roils.sh test all 4586 1 16 1
```

For more information, please see the inline documentation in the code.
