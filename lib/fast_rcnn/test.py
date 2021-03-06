# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob,prep_im_and_label_for_blob,label_list_to_blob
import os
import PIL
from datetime import datetime
import pdb
from score import fast_hist
#from __future__ import print_function


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        #import pdb;pdb.set_trace()
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_image_label_blob(im, label):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order
        label (ndarray): the pixel-wise label of im

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        label_blob (ndarray): a label blob holding resized labels
    """
    im_orig = im.astype(np.float32, copy=True)
    label_orig = label.astype(np.uint8, copy=True)

    processed_ims = []
    processed_labels = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im, im_scale, label = prep_im_and_label_for_blob(im, label, cfg.PIXEL_MEANS, target_size, cfg.TEST.MAX_SIZE)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
        processed_labels.append(label)
        # recover from the original img and label
        im = np.copy(im_orig)
        label = np.copy(label_orig)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    label_blob = label_list_to_blob(processed_labels)

    return blob, np.array(im_scale_factors), label_blob

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois, label=None):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    if label is not None:
        # add img_labels if has labels
        blobs['img_labels'] = label
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

#def _get_blobs_label(im, label):
#    blobs = {'data' : None, 'img_labels': None}
#    blobs['data'], im_scale_factors, blobs['img_labels'] = _get_image_label_blob(im, label)
#    return blobs, im_scale_factors

def im_seg(net, im, label=None):
    """
        Arguments:
            net (caffe.Net): Fast R-CNN network to use
            im (ndarray): color image to test (in BGR order)
            label(ndarray): pixel-wise label for the input image
        Returns:
            scores (ndarray): pixel-wise prediction in object proposals
            boxes (ndarray): R x (4*K) array of predicted bounding boxes
            boxes_score (ndarray): R x 1 array of scores for the predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, None, label)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        # blobs['im_info']: H x W x scale_factor(transform the input image to
        # canonical size)
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if label is not None:
        net.blobs['img_labels'].reshape(*(blobs['img_labels'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if label is not None:
        forward_kwargs['img_labels'] = blobs['img_labels']
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # DEBUG
        #if blobs['img_labels'] > 0:
        #    import pdb;pdb.set_trace()
        # unscale back to raw image space
        #boxes = rois[:, 1:5] * 16 / im_scales[0] # remove after fix
        boxes = rois[:, 1:5] / im_scales[0] # now the change of rois has been fixed
        #import pdb;pdb.set_trace()
        boxes_score = net.blobs['rois_score'].data
    scores = net.blobs['score'].data # (1, 21, 562, 1000), resized image,

    # pick the label for maximum class
    scores = np.argmax(scores, axis=1).astype(np.uint8)
    # resize this scores to map back to the original size of the image
    # only resize if img had diff size
    if im_scales[0] != 1:
        scores = cv2.resize(scores[0,:,:], None, None, fx=1./im_scales[0], fy=1./im_scales[0],
                    interpolation=cv2.INTER_NEAREST)

    return scores, boxes, boxes_score


def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        # blobs['im_info']: H x W x scale_factor(transform the input image to
        # canonical size)
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG: # Default
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

def test_net_seg(net, imdb, suffix='',bbox=None):
    """Test a RPN-FCN network on an image database."""
    print('Evaluating segmentations')
    # This is for saving the segmentation prediction to disk
    #do_seg_tests(net, 0, os.path.join(imdb.data_path, 'pred{}'), imdb, suffix)
    # This is for testing for sorted dice on individual slices
    #do_seg_tests_on_slices(net, 0, False, imdb, suffix)
    # This is for not saving the pred
    #do_seg_tests(net, 0, False, imdb, suffix)
    if bbox is not None:
        do_seg_tests(net, 0, False, imdb, suffix, save_bbox = os.path.join(imdb.data_path,
            'bbox_pred'))
    else:
        do_seg_tests(net, 0, False, imdb, suffix)

def do_seg_tests_on_slices(net, iter, save_format, imdb, suffix, layer='score', gt='label',
        save_bbox = False):
    compute_hist_imdb_for_each_slice(net, save_format, imdb, layer, gt, save_bbox)

def do_seg_tests(net, iter, save_format, imdb, suffix, layer='score', gt='label',
        save_bbox = False):
    if save_format:
        if 'socket' in imdb.name:
            save_format = save_format.format('_socket_'+suffix)
        else:
            save_format = save_format.format(str(iter)+'_'+suffix)
    hist = compute_hist_imdb(net, save_format, imdb, layer, gt, save_bbox)
    #compute_metrics(hist, suffix)
    compute_metrics_flat(hist, suffix)
    #compute_mean_metrics(hist, suffix)
    return hist

def compute_mean_metrics(hist, iter):
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    # per-class Dice Score / F1, dice = 2*iu/(1+iu)
    dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean Dice Score', \
            np.nanmean(dice)

def compute_metrics(hist, iter):
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # positive-class accuracy
    acc = np.diag(hist)[1] / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'accuracy', acc
    # positive-class IU
    iu = np.diag(hist)[1] / (hist[:,1].sum() + hist[1,:].sum() - np.diag(hist)[1])
    print '>>>', datetime.now(), 'Iteration', iter, 'IU', iu
    #freq = hist.sum(1) / hist.sum()
    #print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
    #        (freq[freq > 0] * iu[freq > 0]).sum()
    # positive class Dice Score / F1, dice = 2*iu/(1+iu)
    dice = 2 * np.diag(hist)[1] / (hist[:,1].sum() + hist[1,:].sum())
    print '>>>', datetime.now(), 'Iteration', iter, 'Dice Score', \
            dice
    dice0 = 2*iu/(1+iu)
    print '>>>', datetime.now(), 'Iteration', iter, 'Dice Score from IU', \
            dice0

def compute_metrics_flat(hist, iter):
    eps = 1e-10
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    # recall
    recall = np.diag(hist)[1] / (hist[1,:].sum() + eps)
    # precision
    prec = np.diag(hist)[1] / (hist[:,1].sum() + eps)
    # mean IOU
    #iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    iu = np.diag(hist)[1] / (hist[:,1].sum() + hist[1,:].sum() -
            np.diag(hist)[1] + eps)
    # positive class Dice Score / F1, dice = 2*iu/(1+iu)
    if hist.shape[0] == 2:
        dice = 2 * np.diag(hist)[1] / (hist[:,1].sum() + hist[1,:].sum() + eps)
        print('>>>{}, Iteration {}, accuracy: {}, precision: {}, recall: {}, IOU: {}, Dice: {}'.format(datetime.now(), iter, acc, prec, recall, iu, dice))
    else:
        dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0) + eps)
        print('>>>{}, Iteration {}, accuracy: {}, precision: {}, recall: {}, mean IOU: {}, mean Dice: {}'.format(datetime.now(), iter, acc, prec, recall, np.nanmean(iu), np.nanmean(dice)))

def compute_hist_imdb(net, save_dir, imdb, layer='score', gt='label',
        save_bbox_dir = False,
        loss_layer='loss_cls'):
    n_cl = net.blobs[layer].channels
    if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if save_bbox_dir and not os.path.exists(save_bbox_dir):
        os.mkdir(save_bbox_dir)
    if save_dir:
        # show where to save the image
        print("Pred will be saved to {}".format(save_dir))
    if save_bbox_dir:
        print("Predicted bbox will be saved to {}".format(save_bbox_dir))
    hist = np.zeros((n_cl, n_cl))
    #output_dir = get_output_dir(imdb, net)
    # timers
    _t = {'im_seg' : Timer()}
    num_images = len(imdb.image_index)
    is_jpg = '.jpg' in imdb.image_path_at(0) or '.png' in imdb.image_path_at(0)
    #import pdb;pdb.set_trace()
    for i in xrange(num_images):
        if is_jpg:
            # Load the demo image
            im = cv2.imread(imdb.image_path_at(i))
            # Load the label, in jpeg format
            #label = imdb.get_label(i)
            label = np.asarray(PIL.Image.open(imdb.label_path_at(i)))
            # binarize the label
            #label = np.where(label > 127, 1, 0)
        else:
            # h5 file
            vol_name, sliceidx = imdb.image_path_at(i).rsplit('_',1)
            im = imdb.get_image(vol_name, int(sliceidx))
            im = np.dstack((im, im, im))
            # this label is already binary
            label = imdb.get_label(vol_name, int(sliceidx))

        # One forward pass
        _t['im_seg'].tic()
        im_pred, boxes, boxes_score = im_seg(net, im, label)
        _t['im_seg'].toc()

        if i % 100 == 0:
            print 'im_seg: {:d}/{:d} {:.3f}s' \
                .format(i + 1, num_images, _t['im_seg'].average_time)

        hist += fast_hist(label.flatten(), im_pred.flatten(), n_cl)

        if save_dir:
            # from {0,1} to {0,255} for image display purpose
            im = PIL.Image.fromarray(im_pred*255, mode='L')
            im.save(os.path.join(save_dir, imdb.image_index[i] + '.png'))
        if save_bbox_dir:
            vol_name, sliceidx = imdb.image_path_at(i).rsplit('_',1)
            with open(os.path.join(save_bbox_dir, vol_name+'.txt'),"a") as text_file:
                # format of the line:
                # volname_sliceidx, num_of_boxes, 
                num_box = boxes.shape[0]
                #pdb.set_trace()
                text_file.write("{}_{},{}".format(vol_name,sliceidx,num_box))
                for box_id in xrange(num_box):
                    text_file.write(",{},{},{},{},{}".format(boxes_score[box_id,0],
                        boxes[box_id,0],boxes[box_id,1], boxes[box_id,2],
                        boxes[box_id,3]))
                text_file.write("\n")
        # compute the loss as well
        #loss += net.blobs[loss_layer].data.flat[0]
    if save_dir:
        hist_dir = os.path.join(save_dir, 'hist.txt')
    else:
        hist_dir = os.path.join(imdb.data_path, 'hist.txt')
    with open(hist_dir, 'w') as hist_file:
        np.savetxt(hist_file, hist, '%f')
    return hist

def compute_hist_imdb_for_each_slice(net, save_dir, imdb, layer='score', gt='label',
        save_bbox_dir = False,
        loss_layer='loss_cls'):
    n_cl = net.blobs[layer].channels
    if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if save_dir:
        # show where to save the image
        print("Pred will be saved to {}".format(save_dir))
    # timers
    _t = {'im_seg' : Timer()}
    num_images = len(imdb.image_index)
    for i in xrange(num_images):
        # h5 file
        vol_name, sliceidx = imdb.image_path_at(i).rsplit('_',1)
        im = imdb.get_image(vol_name, int(sliceidx))
        im = np.dstack((im, im, im))
        # this label is already binary
        label = imdb.get_label(vol_name, int(sliceidx))

        # One forward pass
        _t['im_seg'].tic()
        im_pred, boxes, boxes_score = im_seg(net, im, label)
        _t['im_seg'].toc()

        if i % 100 == 0:
            print 'im_seg: {:d}/{:d} {:.3f}s' \
                .format(i + 1, num_images, _t['im_seg'].average_time)

        hist = fast_hist(label.flatten(), im_pred.flatten(), n_cl)
        compute_metrics_flat(hist.astype(np.float32),
                "RPNFCN-{}-{}-{}".format(i, vol_name, sliceidx))

        if save_dir:
            # from {0,1} to {0,255} for image display purpose
            im = PIL.Image.fromarray(im_pred*255, mode='L')
            im.save(os.path.join(save_dir, imdb.image_index[i] + '.png'))
    return hist

