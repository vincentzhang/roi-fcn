# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.hip import hip
from datasets.hip import hiph5
from datasets.socket import socket
from datasets.acce import acce
from datasets.ins import ins
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2011', '2012']:
    for split in ['train', 'val', 'trainval', 'test', 'seg11valid']:
        for category in ['Main', 'Segmentation']:
            if category == 'Main':
                name = 'voc_{}_{}'.format(year, split)
                __sets[name] = (lambda split=split, year=year: pascal_voc(split,
                                year))
            else:
                name = 'voc_{}_{}_{}'.format(year, split, category)
                __sets[name] = (lambda split=split, year=year: pascal_voc(split,
                                year, category))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up hip_<split>
for split in ['train', 'test']:
    name = 'hip_{}'.format(split)
    __sets[name] = (lambda split=split: hip(split))

# Set up socket_<split>
for split in ['train', 'test']:
    for subset in ['partial', 'all']:
        name = 'socket_{}_{}'.format(split, subset)
        use_empty = False if subset == 'partial' else True
        __sets[name] = (lambda split=split, use_empty=use_empty: socket(split, use_empty))

# Set up acce_<split>
for split in ['train', 'test']:
    for subset in ['partial', 'all']:
        name = 'acce_{}_{}'.format(split, subset)
        use_empty = False if subset == 'partial' else True
        __sets[name] = (lambda split=split, use_empty=use_empty: acce(split, use_empty))


# Set up ins_<split>
for split in ['train', 'test']:
    for subset in ['partial', 'all']:
        name = 'ins_{}_{}'.format(split, subset)
        use_empty = False if subset == 'partial' else True
        __sets[name] = (lambda split=split, use_empty=use_empty: ins(split, use_empty))


# Set up hiph5_<split>
for split in ['train', 'test']:
    for subset in ['partial', 'all']:
        name = 'hiph5_{}_{}'.format(split, subset)
        use_empty = False if subset == 'partial' else True
        __sets[name] = (lambda split=split, use_empty=use_empty: hiph5(split, use_empty))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
