from __future__ import division

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, build_sampler,
                        polybox2result,  roi2droi, gt_mask_bp_obbs_list,
                        choose_best_Rroi_batch, down_mask_list, get_seg_masks_with_boxes,
                        aligned_bilinear)

from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS

@DETECTORS.register_module
class BoxLevelset(BaseDetectorNew, RPNTestMixin):

    """
    Deep Levelset for Box-supervised Instance Segmentation in Aerial Images.
    arXiv:2112.03451
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 shared_head_rbbox=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 boxseg_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        assert backbone is not None
        assert neck is not None
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert boxseg_head is not None

        super(BoxLevelset, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if shared_head_rbbox is not None:
            self.shared_head_rbbox = builder.build_shared_head(shared_head_rbbox)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if boxseg_head is not None:
            self.boxseg_head = builder.build_head(boxseg_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.rbbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(BoxLevelset, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox:
            self.shared_head_rbbox.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_boxseg:
            # self.rbbox_roi_extractor.init_weights()
            self.boxseg_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):

        x = self.extract_feat(img)
        losses = dict()
        # trans gt_masks to gt_obbs
        gt_obbs = gt_mask_bp_obbs_list(gt_masks) #gt_masks: gt_poly_mask
        down_gt_poly_masks = down_mask_list(gt_masks) # down-sample gt poly mask

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals (hbb assign)
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn[0].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[0].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            rbbox_targets = self.bbox_head.get_target(
                sampling_results, gt_masks, gt_labels, self.train_cfg.rcnn[0])

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *rbbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(0, name)] = (value)

        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        roi_labels = rbbox_targets[0]
        with torch.no_grad():
            rotated_proposal_list = self.bbox_head.refine_rbboxes(
                roi2droi(rois), roi_labels, bbox_pred, pos_is_gts, img_meta)

        # assign gts and sample proposals (rbb assign)
        if self.with_boxseg:
            bbox_assigner = build_assigner(self.train_cfg.rcnn[1].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[1].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                gt_obbs_best_roi = choose_best_Rroi_batch(gt_obbs[i])
                assign_result = bbox_assigner.assign(
                    rotated_proposal_list[i], gt_obbs_best_roi, gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    rotated_proposal_list[i],
                    torch.from_numpy(gt_obbs_best_roi).float().to(rotated_proposal_list[i].device),
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        if self.with_boxseg:
            rrois = dbbox2roi([res.bboxes for res in sampling_results])
            # feat enlarge
            rrois[:, 3] = rrois[:, 3] * self.boxseg_head.w_enlarge
            rrois[:, 4] = rrois[:, 4] * self.boxseg_head.h_enlarge

            cls_score, init_rbbox_pred, refine_rbbx_pred_list, mask_logits = self.boxseg_head(
                x[:self.boxseg_head.num_level], rrois)

            rbbox_targets = self.boxseg_head.get_target_rbbox_mask(sampling_results, down_gt_poly_masks, gt_labels,
                                                             self.train_cfg.rcnn[1])

            norm_images = self.add_images(img, rrois[:, 0])
            loss_rbbox = self.boxseg_head.loss(cls_score, init_rbbox_pred, refine_rbbx_pred_list, mask_logits,
                                              norm_images, img_meta[0]['img_shape'], *rbbox_targets)

            for name, value in loss_rbbox.items():
                losses['s{}.{}'.format(1, name)] = (value)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):

        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        bbox_label = cls_score.argmax(dim=1)
        rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred, img_meta[0])

        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.boxseg_head.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.boxseg_head.h_enlarge

        rcls_score, initial_rbbox_pred, refine_rbbox_pred_list, mask_logits = self.boxseg_head(
            x[:len(self.boxseg_head.featmap_strides)], rrois_enlarge)

        det_rbboxes, det_labels, det_masks = self.boxseg_head.get_det_rbboxes(
            rcls_score,
            # initial_rbbox_pred,
            refine_rbbox_pred_list[-1],
            mask_logits,
            img_meta[0]['img_shape'],
            img_meta[0]['scale_factor'],
            rescale=rescale,
            cfg=self.test_cfg.rcnn)

        masks_results = self.mask_postprocess(det_masks, img_meta[0]['img_shape'])
        masks_results_encode = get_seg_masks_with_boxes(masks_results, det_rbboxes, det_labels,
                                           self.boxseg_head.num_classes, img_meta[0]['img_shape'])

        # poly box results
        # rbbox_results = dbbox2result(det_rbboxes, det_labels,
        #                              self.boxseg_head.num_classes)
        #

        #hbb box results for map evaluation
        bbox_results = polybox2result(det_rbboxes, det_labels,
                                     self.boxseg_head.num_classes) # hbb for evaluation

        # return rbbox_results, masks_results_encode
        return bbox_results, masks_results_encode

    def add_images(self, img, im_idx, stride=4):
        '''
        add the corresponding norm images for levelset evolution
        '''
        stride = stride
        downsampled_images = F.interpolate(
            img.float(), size=(int(img.size(2) / stride), int(img.size(3) / stride)),
            mode='bilinear',
            align_corners=True
        )
        batch_size = img.size(0)
        sampled_imgs = torch.zeros([im_idx.size(0),
                                    img.size(1), #3
                                    downsampled_images.size(2),
                                    downsampled_images.size(3)]).to(im_idx.device) #(N, 3, W/8, H/8)

        for i in range(batch_size):
            sampled_imgs[im_idx == i] = downsampled_images[i:i + 1]
        return sampled_imgs

    def mask_postprocess(self, mask_logits_preds, img_shape):
        """
        mask potsprocess based on the predicted mask logits.
        """
        (img_h, img_w, img_chan) = img_shape
        if mask_logits_preds.size(0) > 0:

            mask_h, mask_w = mask_logits_preds.size()[-2:]
            factor_h = img_h // mask_h
            factor_w = img_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                mask_logits_preds, factor
            )

            pred_global_masks = pred_global_masks[:, :, :img_h, :img_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(img_h, img_w),
                mode="bilinear", align_corners=False
            )
            pred_masks = pred_global_masks[:, 0, :, :].float()
        else:
            pred_masks = torch.zeros(0, img_h, img_w)

        return pred_masks



