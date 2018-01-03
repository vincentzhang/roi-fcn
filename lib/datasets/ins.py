# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import PIL
import h5py
import cv2
import pdb


class ins(imdb):
    def __init__(self, image_set, use_empty=False, devkit_path=None, vol=None):
        imdb.__init__(self, 'ins_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = self._devkit_path
        # image_set = train or test
        # self._devkit_path: data/ins/
        # self._data_path  : data/ins/
        # train image: data/ins/train.txt
        # test image: data/ins/test.txt
        # bbox: data/ins/bbox
        # image: data/ins/images
        # pix-label: data/ins/labels
        self._classes = ('background', # always index 0
                         'foreground')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._vol_names = self._get_vol_names()
        #self._vol_names = self._get_vol_names(vol="Me6401IM_0063")#"So2601IM_0022")
        self._image_ext = '.png'
        self._label_ext = '.png'
        # ins specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'matlab_eval' : False,
                       'use_empty'   : False}
        # use_empty: True to use the empty slices that do not contain bboxes
        #if self._image_set == 'test' or use_empty:
        if use_empty:
            self.config['use_empty'] = True

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        assert os.path.exists(self._devkit_path), \
                'socket path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    @property
    def add_label(self):
        ret = True
        return ret

    @property
    def data_path(self):
        return self._data_path

    def _get_vol_names(self, vol=None):
        if vol:
            print("vol is: ", vol)
            return [vol]
        vol_file = os.path.join(self._data_path, self._image_set + '_vol.txt')
        assert os.path.exists(vol_file), \
                'vol path does not exist: {}'.format(vol_file)
        with open(vol_file) as fvol:
            # list of image set name
            return [x.strip() for x in fvol.readlines()]

    def get_size(self):
        """ width, height """
        sizes = [PIL.Image.open(self.image_path_at(i)).size
                for i in xrange(self.num_images)]
        return sizes

    def get_image(self, i):
        return cv2.imread(self.image_path_at(i))

    def get_label(self, i):
        """ Binary set {0,1} """
        return np.asarray(PIL.Image.open(self.label_path_at(i)), dtype='uint8')

    def get_bbox(self, i):
        """ Return a dictionary that contains bbox coordinates """
        return self._load_ins_annotation(self._image_index[i])

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'images',
                'data_'+index+self._label_ext)
        #print("Image is at: ", image_path)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load a list of indexes listed in this dataset's image set file.
        Format: volname_framexxx
        """
        image_index = []
        for name in self._vol_names:
            bbox_file = os.path.join(self._data_path, 'bbox', 'bbox_' + name + '.txt')
            assert os.path.exists(bbox_file), \
                    'bbox path does not exist: {}'.format(bbox_file)
            with open(bbox_file) as fbbox:
                if self.config['use_empty']:
                    vol_index = [x.strip().split(',')[0] for x in fbbox.readlines()]
                else:
                    vol_index = [x.strip().split(',')[0] for x in fbbox.readlines() if x.strip().split(',')[1]!='0']

            image_index += vol_index

        return image_index

    def label_path_at(self, i):
        """
        Return the absolute path to label of image i in the image sequence.
        """
        return self._label_path_from_index(self._image_index[i])

    def _label_path_from_index(self, index):
        """
        Construct an label path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path,
                'labels', 'label_'+index + self._label_ext)
        #print("Label is at: ", label_path)
        assert os.path.exists(label_path), \
                'Label Path does not exist: {}'.format(label_path)
        return label_path

    def _get_default_path(self):
        """
        Return the default path where socket is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'ins')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_ins_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def gt_img_labels(self):
        """
        Return the database of ground-truth pixel-wise labels.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        assert False, 'not supposed to call this function'
        cache_file = os.path.join(self.cache_path, self.name + '_gt_img_labels.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_img_labels = cPickle.load(fid)
            print '{} gt img labels loaded from {}'.format(self.name, cache_file)
            return gt_img_labels

        gt_img_labels = [self._load_hip_annotation_segment(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_img_labels, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt img labels to {}'.format(cache_file)

        return gt_img_labels

    def _load_ins_annotation(self, index):
        # the index is the image name: 1_frame000
        # data: /images/data_1_frame000.png
        # datadir
        # labeldir
        vol_name, sliceidx = index.rsplit('_',1)
        image_set_file = os.path.join(self._data_path,
                'bbox', 'bbox_' + vol_name + '.txt')
        assert os.path.exists(image_set_file), \
                'Loading ins annotations: path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f: # only keep the numbers in the middle
            for x in f:
                # 1_frame000,3,170,5,345,405,94,353,174,560,59,522,178,634
                text = x.strip().split(',')
                if text[0] != index:
                    # if not the slice, continue to the next
                    continue
                # text: vol_sliceidx, num_bbox, x1,y1,x2,y2
                num_objs = int(text[1])
                if not self.config['use_empty']:
                    assert num_objs != 0 # only slices with bbox should be used
                boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_objs), dtype=np.int32)
                overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
                # "Seg" area for socket data is just the box area
                seg_areas = np.zeros((num_objs), dtype=np.float32)

                # Load object bounding boxes into a data frame.
                for ix in range(num_objs):
                    # 0-based, x1,y1,x2,y2
                    x1 = float(text[2+ix*4])
                    y1 = float(text[3+ix*4])
                    x2 = float(text[4+ix*4])
                    y2 = float(text[5+ix*4])
                    #if (x1 == x2) and (y1 == y2): # remove entries with area of 1
                    #    continue
                    boxes[ix, :] = [x1, y1, x2, y2]
                    gt_classes[ix] = 1
                    overlaps[ix, 1] = 1.0
                    seg_areas[ix] = (boxes[ix,2] - boxes[ix, 0] + 1) * (boxes[ix,3] - boxes[ix,1] + 1)
                # check if any area is 1
                mask = np.ones(num_objs, dtype=bool)
                for obj_id, area in enumerate(seg_areas):
                    if area <= 5:
                        mask[obj_id] = False
                boxes = boxes[mask, :]
                gt_classes = gt_classes[mask]
                overlaps = overlaps[mask, :] 
                seg_areas = seg_areas[mask]
                break

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def evaluate_seg(self, output_dir):
        seg_file = os.path.join(output_dir, 'seg.pkl')
        with open(seg_file, 'wb') as f:
            cPickle.dump(all_seg, f, cPickle.HIGHEST_PROTOCOL)
        # flatten the images and compute the IOU

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
