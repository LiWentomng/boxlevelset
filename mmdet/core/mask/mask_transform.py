import torch
import numpy as np
import mmcv
import pycocotools.mask as mask_util
import torch.nn.functional as F

def poly2mask_single(poly, h=800, w=800):

    # visualize the mask
    rles = mask_util.frPyObjects(poly, h, w)
    rle = mask_util.merge(rles)
    mask = mask_util.decode(rle)

    return mask

def get_seg_masks_with_boxes(mask_pred, det_boxes, det_labels, num_classes, img_shape):
    """Get segmentation masks from mask_pred within bboxes.
    Args:
        mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
            For single-scale testing, mask_pred is the direct output of
            model, whose type is Tensor, while for multi-scale testing,
            it will be converted to numpy array outside of this method.
        det_boxes (Tensor): shape (n, 4/5)
        det_labels (Tensor): shape (n, )

    Returns:
        list[list]: encoded masks
    """
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu().numpy()
    if isinstance(det_boxes, torch.Tensor):
        det_boxes = det_boxes.cpu().numpy()
    assert isinstance(mask_pred, np.ndarray)
    assert isinstance(det_boxes, np.ndarray)

    (img_h, img_w, img_cha) = img_shape
    cls_segms = [[] for _ in range(num_classes - 1)]
    labels = det_labels.cpu().numpy() + 1

    for i in range(mask_pred.shape[0]):
        label = labels[i]
        mask_threshold = get_seg_threshold(label)
        det_boxes_ = det_boxes[i][0:8]

        # if label == 15: #helicopter
        #     mask_pred_ = 1.0 - ((mask_pred[i, :, :] > mask_threshold).astype(mask_pred.dtype))
        # else:
        mask_pred_ = (mask_pred[i, :, :] > mask_threshold).astype(mask_pred.dtype)
        det_poly_mask = poly2mask_single([det_boxes_], img_h, img_w).astype(mask_pred_.dtype)

        im_mask = (mask_pred_ * det_poly_mask).astype(np.uint8)
        rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        cls_segms[label - 1].append(rle)

    return cls_segms


def get_seg_masks(mask_pred, det_labels, num_classes):
    """Get segmentation masks from mask_pred and bboxes.

    Args:
        mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
            For single-scale testing, mask_pred is the direct output of
            model, whose type is Tensor, while for multi-scale testing,
            it will be converted to numpy array outside of this method.
        det_bboxes (Tensor): shape (n, 4/5)
        det_labels (Tensor): shape (n, )

    Returns:
        list[list]: encoded masks
    """
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu().numpy()
    assert isinstance(mask_pred, np.ndarray)

    cls_segms = [[] for _ in range(num_classes - 1)]
    labels = det_labels.cpu().numpy() + 1

    for i in range(mask_pred.shape[0]):

        label = labels[i]
        mask_pred_ = mask_pred[i, :, :]
        im_mask = mask_pred_.astype(
            np.uint8)

        rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        cls_segms[label - 1].append(rle)

    return cls_segms

def get_seg_threshold(det_label):
    '''
    CLASSES = ('plane 1 ', 'baseball-diamond 2',
               'bridge 3', 'ground-track-field 4',
               'small-vehicle 5', 'large-vehicle 6',
               'ship 7', 'tennis-court 8',
               'basketball-court 9', 'storage-tank 10',
               'soccer-ball-field 11', 'roundabout 12',
               'harbor 13', 'swimming-pool 14',
               'helicopter 15')
    '''
    if det_label == 4:  # plane
        threshold = 0.20
    elif det_label == 15:  # helicopter
        threshold = 0.15
    elif det_label == 7:  # harbor
        threshold = 0.10
    elif det_label == 8 or det_label == 10 or det_label == 13:  # tennis  soccerball  basketball
        threshold = 0.01
    elif det_label == 6:  # swimming pool
        threshold = 0.02
    elif det_label == 3 or det_label == 1:  # large_vehicle small vehicle
        threshold = 0.03
    else:
        threshold = 0.05
    return threshold

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )
    return tensor[:, :, :oh - 1, :ow - 1]


def get_horizontal_bboxes(gt_rbboxes):
    gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
    gt_xmin, _ = gt_xs.min(1)
    gt_ymin, _ = gt_ys.min(1)
    gt_xmax, _ = gt_xs.max(1)
    gt_ymax, _ = gt_ys.max(1)
    gt_rect_bboxes = torch.cat([gt_xmin[:, None], gt_ymin[:, None],
                                gt_xmax[:, None], gt_ymax[:, None]], dim=1)

    return gt_rect_bboxes


def get_enlarge_horizontal_bboxes(gt_rbboxes, img_h, img_w):
    gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]

    gt_xmin, _ = gt_xs.min(1)
    gt_ymin, _ = gt_ys.min(1)
    gt_xmax, _ = gt_xs.max(1)
    gt_ymax, _ = gt_ys.max(1)
    gt_center_x = (gt_xmin + gt_xmax) / 2.0
    gt_center_y = (gt_ymin + gt_ymax) / 2.0
    gt_half_w = (gt_xmax - gt_xmin) / 2.0
    gt_half_h = (gt_ymax - gt_ymin) / 2.0

    enlarge_gt_xmin = (gt_center_x - 2.0 * gt_half_w).clamp(min=0.0)
    enlarge_gt_ymin = (gt_center_y - 2.0 * gt_half_h).clamp(min=0.0)
    enlarge_gt_xmax = (gt_center_x + 2.0 * gt_half_w).clamp(max=img_w)
    enlarge_gt_ymax = (gt_center_y + 2.0 * gt_half_h).clamp(max=img_h)

    gt_enlarge_rect_bboxes = torch.cat([enlarge_gt_xmin[:, None], enlarge_gt_ymin[:, None],
                                enlarge_gt_xmax[:, None], enlarge_gt_ymax[:, None]], dim=1)

    return gt_enlarge_rect_bboxes


def add_bitmasks_from_boxes(poly_boxes, im_h, im_w):
    '''
    im_h:  images h
    im_w:  images w
    '''
    stride = 4
    start = int(stride // 2)
    bboxes = get_horizontal_bboxes(poly_boxes)
    # enlarge_bboxes = self.get_enlarge_horizontal_bboxes(poly_boxes)

    per_box_bitmasks = []

    for per_box in bboxes:
        bitmask_full = torch.zeros((im_h, im_w)).to(poly_boxes.device).float()
        bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

        bitmask = bitmask_full[start::stride, start::stride]

        assert bitmask.size(0) * stride == im_h
        assert bitmask.size(1) * stride == im_w

        per_box_bitmasks.append(bitmask)
        # per_box_bitmasks_full.append(bitmask_full)

    gt_bitmasks = torch.stack(per_box_bitmasks, dim=0)
    return gt_bitmasks


def add_bitmasks_from_enlarge_boxes(poly_boxes, im_h, im_w):
    '''
    im_h:  images h
    im_w:  images w
    '''
    stride = 4
    start = int(stride // 2)
    enlarge_bboxes = get_enlarge_horizontal_bboxes(poly_boxes, im_h, im_w)
    per_box_bitmasks = []

    for per_box in enlarge_bboxes:
        bitmask_full = torch.zeros((im_h, im_w)).to(poly_boxes.device).float()
        bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

        bitmask = bitmask_full[start::stride, start::stride]
        assert bitmask.size(0) * stride == im_h
        assert bitmask.size(1) * stride == im_w
        per_box_bitmasks.append(bitmask)

    gt_bitmasks = torch.stack(per_box_bitmasks, dim=0)

    return gt_bitmasks