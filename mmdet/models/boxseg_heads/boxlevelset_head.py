import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import delta2dbbox, multiclass_nms_rbbox, \
    bbox_target_rbbox, accuracy, \
    delta2dbbox_v3, hbb2obb_v2, multiclass_nms_polybbox, \
    multiclass_nms_polybbox_mask, rbbox_target_poly_mask

from mmdet.ops.minarearect import minaerarect
from ..builder import build_loss
from ..registry import HEADS
from mmdet.core import RotBox2Polys_torch
from ..utils import _SnakeNet, ASPP_module, get_class_levelset_weight, get_class_enlarge_num

@HEADS.register_module
class BoxLevelsetHead(nn.Module):
    """
    Detection and Instance Segementation Head for Deep Levelset Evolution.
    arXiv:2112.03451
    """

    def __init__(self,
                 conv_dims=256,
                 num_convs=2,
                 dilations=(1, 1),
                 num_iter=(1,),
                 num_classes=15,
                 num_fcs=2,
                 fc_out_channels=1024,
                 featmap_strides=[4, 8, 16, 32],
                 with_module=True,
                 hbb_trans='hbb2obb_v2',
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_bbox_refine=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_seg_mask = dict(type='LevelsetLoss')):
        super(BoxLevelsetHead, self).__init__()

        self.conv_dims = conv_dims
        self.num_convs = num_convs
        self.dilations = dilations
        self.num_iter = num_iter
        # self.reg_class_agnostic = reg_class_agnostic
        self.num_classes = num_classes
        self.num_fcs = num_fcs
        self.fc_out_channels = fc_out_channels

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_seg_mask = build_loss(loss_seg_mask)

        self.with_module = with_module
        self.hbb_trans = hbb_trans

        self.target_means = [0., 0., 0., 0., 0.]
        self.target_stds = [0.05, 0.05, 0.1, 0.1, 0.05]

        self.w_enlarge = 1.2
        self.h_enlarge = 1.4
        self.featmap_strides = featmap_strides
        self.num_level = len(featmap_strides)

        self.rates = [1, 6, 12, 18]
        self.register_buffer("_iter", torch.zeros([1]))
        self._warmup_iters = 9500

        self._init_layers()

    def _init_layers(self):
        # feature[p2]
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AvgPool2d(7)
        self.fc_cls_branch = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                (self.conv_dims+2)*40 if i == 0 else self.fc_out_channels)
            self.fc_cls_branch.append(
                nn.Linear(fc_in_channels, self.fc_out_channels))

        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)
        self.aspp1 = ASPP_module(256, 256, rate=self.rates[0])
        self.aspp2 = ASPP_module(256, 256, rate=self.rates[1])
        self.aspp3 = ASPP_module(256, 256, rate=self.rates[2])
        # self.aspp4 = ASPP_module(256, 256, rate=self.rates[3])

        # self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)
        self.conv1 = nn.Conv2d(768, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.bottom_out = nn.Conv2d(
            self.conv_dims,
            1,
            3,
            padding=1)
        self.fuse = nn.Conv1d(2 * self.conv_dims, self.conv_dims, 1)
        self.init_snake = _SnakeNet(state_dim=128, feature_dim=self.conv_dims + 2)
        for i in range(len(self.num_iter)):
            snake_deformer = _SnakeNet(state_dim=128, feature_dim=self.conv_dims + 2)
            self.__setattr__('deformer' + str(i), snake_deformer)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _init_regression(self, deformer, features, locations, image_idx):
        '''
        init regression in detection branch
        '''
        sampled_features = self.get_locations_feature(features, locations, image_idx)

        center = (torch.min(locations, dim=1)[0] + torch.max(locations, dim=1)[0]) * 0.5
        center_feat = self.get_locations_feature(features, center[:, None], image_idx)
        init_feat = torch.cat([sampled_features, center_feat.expand_as(sampled_features)], dim=1)
        init_feat = self.fuse(init_feat)
        calibrated_locations = self.de_location(locations)

        concat_features = torch.cat([init_feat, calibrated_locations.permute(0, 2, 1)], dim=1)  #
        pred_offsets = deformer(concat_features).permute(0, 2, 1)
        init_pred_locations = locations + pred_offsets
        return init_pred_locations, concat_features# N, 258, 40

    def _refine_regression(self, deformer, features, locations, image_idx):
        """
        refine regression for detection branch.
        """
        sampled_features = self.get_locations_feature(features, locations, image_idx)
        calibrated_locations = self.de_location(locations)
        concat_features = torch.cat([sampled_features, calibrated_locations.permute(0, 2, 1)], dim=1)
        pred_offsets = deformer(concat_features).permute(0, 2, 1)
        pred_locations = locations + pred_offsets
        return pred_locations

    def sampling_points(self, corners, points_num):
        '''
        sample the edge points for snake module in the detection branch
        param corners: the corner points of the oriented bounding box.
        param points_num: the number of samping points.
        return: the sampling points of the whole oriented bounding box.
        '''
        device = corners.device
        corners_xs, corners_ys = corners[:, 0::2], corners[:, 1::2]

        edge1_pts_x, edge1_pts_y = corners_xs[:, 0:2], corners_ys[:, 0:2]
        edge2_pts_x, edge2_pts_y = corners_xs[:, 1:3], corners_ys[:, 1:3]
        edge3_pts_x, edge3_pts_y = corners_xs[:, 2:4], corners_ys[:, 2:4]
        edge4_pts_s_x, edge4_pts_s_y = corners_xs[:, 3], corners_ys[:, 3]
        edge4_pts_e_x, edge4_pts_e_y = corners_xs[:, 0], corners_ys[:, 0]

        # sampling ratio
        ratio = torch.linspace(0, 1, points_num).to(device).repeat(corners.shape[0], 1)
        pts_edge1_x = ratio * edge1_pts_x[:, 1:2] + (1 - ratio) * edge1_pts_x[:, 0:1]
        pts_edge1_y = ratio * edge1_pts_y[:, 1:2] + (1 - ratio) * edge1_pts_y[:, 0:1]
        pts_edge2_x = ratio * edge2_pts_x[:, 1:2] + (1 - ratio) * edge2_pts_x[:, 0:1]
        pts_edge2_y = ratio * edge2_pts_y[:, 1:2] + (1 - ratio) * edge2_pts_y[:, 0:1]
        pts_edge3_x = ratio * edge3_pts_x[:, 1:2] + (1 - ratio) * edge3_pts_x[:, 0:1]
        pts_edge3_y = ratio * edge3_pts_y[:, 1:2] + (1 - ratio) * edge3_pts_y[:, 0:1]
        pts_edge4_x = ratio * edge4_pts_e_x.unsqueeze(1) + (1 - ratio) * edge4_pts_s_x.unsqueeze(1)
        pts_edge4_y = ratio * edge4_pts_e_y.unsqueeze(1) + (1 - ratio) * edge4_pts_s_y.unsqueeze(1)

        pts_x = torch.cat([pts_edge1_x, pts_edge2_x, pts_edge3_x, pts_edge4_x], dim=1).unsqueeze(dim=2)
        pts_y = torch.cat([pts_edge1_y, pts_edge2_y, pts_edge3_y, pts_edge4_y], dim=1).unsqueeze(dim=2)

        sampling_poly_points = torch.cat([pts_x, pts_y], dim=2)

        return sampling_poly_points


    def get_locations_feature(self, features, locations, image_idx):
        '''
        :param feat: list like [(b, c, h/s, w/s), ...]
        :param img_poly: (Sigma{num_poly_i}, poly_num, 2) - scaled by s
        :param ind: poly corresponding index to image
        :param lvl: list, poly corresponding index to feat level
        :return:
        '''
        h = features.shape[2] * 4
        w = features.shape[3] * 4
        locations = locations.clone()
        locations[..., 0] = locations[..., 0] / (w / 2.) - 1
        locations[..., 1] = locations[..., 1] / (h / 2.) - 1

        batch_size = features.size(0)
        sampled_features = torch.zeros([locations.size(0), #b
                                          features.size(1), #c
                                          locations.size(1)]).to(locations.device) #num_points
        for i in range(batch_size):
            per_im_loc = locations[image_idx == i].unsqueeze(0)
            feature = torch.nn.functional.grid_sample(features[i:i + 1], per_im_loc)[0].permute(1, 0, 2)
            sampled_features[image_idx == i] = feature

        return sampled_features

    def get_mask_feature(self, features, image_idx):
        '''
        :param feat: list like [(b, c, h/s, w/s), ...]
        :param img_poly: (Sigma{num_poly_i}, poly_num, 2) - scaled by s
        :param ind: poly corresponding index to image
        :param lvl: list, poly corresponding index to feat level
        :return:
        '''
        batch_size = features.size(0)
        sampled_features = torch.zeros([image_idx.size(0), #N
                                        features.size(2), #W
                                        features.size(3)]).to(image_idx.device)
        for i in range(batch_size):
            sampled_features[image_idx == i] = features[i:i+1].squeeze(dim=0)

        return sampled_features.unsqueeze(dim=0)

    def de_location(self, locations):
        # de-location :spatial relationship among locations; translation invariant
        x_min = torch.min(locations[..., 0], dim=-1)[0]
        y_min = torch.min(locations[..., 1], dim=-1)[0]
        new_locations = locations.clone()

        # no normalization, this helps maitain the shape I think~
        new_locations[..., 0] = (new_locations[..., 0] - x_min[..., None])
        new_locations[..., 1] = (new_locations[..., 1] - y_min[..., None])
        return new_locations / 32.0

    def forward(self, x, rois):
        '''
        x: FPN features
        rois_location: the output of RPN, args:(x, y, w, h, theat)
        '''
        # regression
        rois_poly = RotBox2Polys_torch(rois[:, 1:])
        poly_idx = rois[:, 0]
        sampling_points = self.sampling_points(rois_poly, 10)
        feats_p2 = x[0]  # B, C, H, W(2, 256, 256, 256)
        initial_pred_locations, _ = self._init_regression(self.init_snake, feats_p2, sampling_points, poly_idx)
        refine_preds = []
        points_feats = []
        for i in range(len(self.num_iter)):
            deformer = self.__getattr__('deformer' + str(i))
            if i == 0:
                refine_pred_locations, points_feat = self._init_regression(deformer, feats_p2, initial_pred_locations, poly_idx)
            else:
                refine_pred_locations, points_feat = self._init_regression(deformer, feats_p2, refine_pred_locations, poly_idx)
            refine_preds.append(refine_pred_locations) #list
            points_feats.append(points_feat)

        # classification
        feat_cls = points_feats[-1].view(points_feats[-1].size(0), -1)
        for fc in self.fc_cls_branch:
            feat_cls = self.relu(fc(feat_cls))
        cls_score = self.fc_cls(feat_cls)

        (_, _, p2_h, p2_w) = x[0].shape
        #mask_branch
        level_p3_feats = F.interpolate(x[1], size=(p2_h, p2_w), mode='bilinear', align_corners=True)
        level_p4_feats = F.interpolate(x[2], size=(p2_h, p2_w), mode='bilinear', align_corners=True)
        level_p5_feats = F.interpolate(x[3], size=(p2_h, p2_w), mode='bilinear', align_corners=True)
        mask_feature = (x[0] + level_p3_feats + level_p4_feats + level_p5_feats) / 4.0

        mask_feats_x1 = self.aspp1(mask_feature)
        mask_feats_x2 = self.aspp2(mask_feature)
        mask_feats_x3 = self.aspp3(mask_feature)

        # mask_feats_concat = torch.cat((mask_feats_x1, mask_feats_x2, mask_feats_x3, mask_feats_x4), dim=1)
        mask_feats_concat = torch.cat((mask_feats_x1, mask_feats_x2, mask_feats_x3), dim=1)
        mask_feats_x = self.conv1(mask_feats_concat)
        mask_feats_x = self.bn1(mask_feats_x)
        mask_feats_x = self.relu(mask_feats_x)

        mask_feats = self.bottom_out(mask_feats_x)
        mask_logits = self.get_mask_feature(mask_feats, poly_idx) # N,1, 200, 200

        mask_logits_scores = mask_logits.reshape(-1, 1, p2_h, p2_w)

        return cls_score, initial_pred_locations, refine_preds, mask_logits_scores

    def get_target(self, sampling_results, gt_masks, gt_labels,
                   rcnn_train_cfg):
        """
        obb target hbb
        :param sampling_results:
        :param gt_masks:
        :param gt_labels:
        :param rcnn_train_cfg:
        :param mod: 'normal' or 'best_match', 'best_match' is used for RoI Transformer
        :return:
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds  for res in sampling_results
        ]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_assigned_gt_inds,
            gt_masks,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            with_module=self.with_module,
            hbb_trans=self.hbb_trans)
        return cls_reg_targets

    def get_target_rbbox_mask(self, sampling_results, gt_masks, gt_labels,
                   rcnn_train_cfg):
        """
        get obb and box mask target
        :param sampling_results:
        :param gt_bboxes:
        :param gt_labels:
        :param rcnn_train_cfg:
        :return:
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = self.num_classes

        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        cls_reg_targets = rbbox_target_poly_mask(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_assigned_gt_inds,
            gt_masks,
            pos_gt_labels,
            rcnn_train_cfg)

        return cls_reg_targets

    def add_class_bitmasks_from_enlarge_boxes(self, poly_boxes, im_h, im_w, labels, pos_inds):
        '''
        im_h:  images h
        im_w:  images w
        '''
        stride = 4
        start = int(stride // 2)
        # bboxes = self.get_horizontal_bboxes(poly_boxes)
        # enlarge_bboxes = self.get_enlarge_horizontal_bboxes(poly_boxes)
        enlarge_bboxes = self.get_class_enlarge_horizontal_bboxes(poly_boxes, labels, pos_inds, im_h, im_w)
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

    def get_class_enlarge_horizontal_bboxes(self, gt_rbboxes, class_labels, pos_inds, img_h, img_w):
        """
        get enlarged hbb box region
        """
        gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
        gt_xmin, _ = gt_xs.min(1)
        gt_ymin, _ = gt_ys.min(1)
        gt_xmax, _ = gt_xs.max(1)
        gt_ymax, _ = gt_ys.max(1)

        gt_center_x = (gt_xmin + gt_xmax) / 2.0
        gt_center_y = (gt_ymin + gt_ymax) / 2.0
        gt_half_w = (gt_xmax - gt_xmin) / 2.0
        gt_half_h = (gt_ymax - gt_ymin) / 2.0
        class_enlarge_num = get_class_enlarge_num(class_labels, pos_inds).squeeze(dim=1)

        enlarge_gt_xmin = (gt_center_x - class_enlarge_num * gt_half_w).clamp(min=0.0)
        enlarge_gt_ymin = (gt_center_y - class_enlarge_num * gt_half_h).clamp(min=0.0)

        enlarge_gt_xmax = (gt_center_x + class_enlarge_num * gt_half_w).clamp(max=img_w)
        enlarge_gt_ymax = (gt_center_y + class_enlarge_num * gt_half_h).clamp(max=img_h)

        gt_enlarge_rect_bboxes = torch.cat([enlarge_gt_xmin[:, None], enlarge_gt_ymin[:, None],
                                    enlarge_gt_xmax[:, None], enlarge_gt_ymax[:, None]], dim=1)

        return gt_enlarge_rect_bboxes

    def compute_project_term(self, mask_scores, gt_bitmasks):
        """
        box mask projection loss
        """
        mask_losses_y = self.dice_coefficient(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = self.dice_coefficient(
            mask_scores.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )
        return (mask_losses_x + mask_losses_y).mean()

    def dice_coefficient(self, x, target):
        """
        dice coefficient for box projection
        """
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def dice_mask_term(self, pred_logits, target_mask):
        """
        dice loss for background constraint.
        """
        eps = 1e-5
        n_inst = pred_logits.size(0)
        pred_logits = pred_logits.reshape(n_inst, -1)
        target_mask = target_mask.reshape(n_inst, -1)
        intersection = (pred_logits * target_mask).sum(dim=1)
        union = (pred_logits ** 2.0).sum(dim=1) + (target_mask ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss.mean()


    def loss(self,
             cls_score,
             init_points_pred,
             refine_points_pred_list,
             mask_logits,
             norm_imgs,
             img_shape,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             mask_targets,
             reduce=True):

        self._iter += 1
        losses = dict()
        losses['rbbox_loss_cls'] = self.loss_cls(
            cls_score, labels, label_weights, reduce=reduce)
        losses['rbbox_acc'] = accuracy(cls_score, labels)

        (img_h, img_w, img_c) = img_shape

        pos_inds = labels > 0
        levelset_class_weight = get_class_levelset_weight(labels, pos_inds).unsqueeze(dim=2).unsqueeze(dim=3)
        pos_init_points_pred = init_points_pred[pos_inds]
        poly_targets = RotBox2Polys_torch(bbox_targets)

        #edge points  for snake regression
        points_targets = self.sampling_points(poly_targets, 10)

        #gt_masks
        enlarge_gt_bitmasks_pos = self.add_class_bitmasks_from_enlarge_boxes(poly_targets[pos_inds], img_h, img_w, labels, pos_inds)  #[62, 200, 200]
        enlarge_gt_bitmasks_pos = enlarge_gt_bitmasks_pos.unsqueeze(dim=1).to(dtype=mask_logits.dtype) #[62, 1, 200, 200]

        losses['init_points_loss'] = self.loss_bbox_init(
                pos_init_points_pred / 8.0,
                points_targets[pos_inds] / 8.0).sum() / bbox_targets.size(0)
        refine_loss = 0.0
        for refine_points_pred in refine_points_pred_list:
            refine_loss += (self.loss_bbox_refine(
                refine_points_pred[pos_inds] / 8.0,
                points_targets[pos_inds] / 8.0).sum() / bbox_targets.size(0))
        # losses['refine_points_loss'] = refine_loss / 3.0
        losses['refine_points_loss'] = refine_loss

        #mask segmentation loss
        pos_mask_logits = mask_logits[pos_inds]
        pos_mask_targets = mask_targets[pos_inds].unsqueeze(dim=1).to(dtype=mask_logits.dtype)

        mask_scores = pos_mask_logits.sigmoid() #(N, 1, 200, 200)
        background_mask_scores = (- pos_mask_logits).sigmoid() # (N, 1, 200, 200)

        # img
        pos_norm_imgs = norm_imgs[pos_inds]
        # box constraint and background constraint
        losses['box_project_loss'] = self.compute_project_term(mask_scores, pos_mask_targets)
        losses['background_dice_loss'] = self.dice_mask_term(background_mask_scores, (1.0 - pos_mask_targets))

        warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
        mask_scores_concat = torch.cat([mask_scores, background_mask_scores], dim=1)

        #enlarged box region
        pos_norm_imgs_enbox_region = pos_norm_imgs * enlarge_gt_bitmasks_pos
        mask_scores_concat_enbox_region = mask_scores_concat * enlarge_gt_bitmasks_pos

        losses['levelset_loss'] = warmup_factor * self.loss_seg_mask(mask_scores_concat_enbox_region, pos_norm_imgs_enbox_region, levelset_class_weight)
        return losses


    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        # TODO: check and simplify it
        if rois.size(1) == 5:
            obbs = hbb2obb_v2(rois[:, 1:])
        elif rois.size(1) == 6:
            obbs = rois[:, 1:]
        else:
            print('strange size')
            import pdb
            pdb.set_trace()
        if bbox_pred is not None:
            if self.with_module:
                dbboxes = delta2dbbox(obbs, bbox_pred, self.target_means,
                                      self.target_stds, img_shape)
            else:
                dbboxes = delta2dbbox_v3(obbs, bbox_pred, self.target_means,
                                         self.target_stds, img_shape)
        else:
            dbboxes = obbs
            # TODO: add clip here

        if rescale:
            dbboxes[:, 0::5] /= scale_factor
            dbboxes[:, 1::5] /= scale_factor
            dbboxes[:, 2::5] /= scale_factor
            dbboxes[:, 3::5] /= scale_factor

        c_device = dbboxes.device
        det_bboxes, det_labels = multiclass_nms_rbbox(dbboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)

        return det_bboxes, det_labels


    def get_det_rbboxes(self,
                       cls_score,
                       refine_rbbox_pred,
                       mask_logits,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):

        (img_h, img_w, img_c) = img_shape

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        poly_rbboxes = minaerarect(refine_rbbox_pred.reshape(refine_rbbox_pred.size(0), 2*40))

        if rescale:
            poly_rbboxes /= scale_factor

        mask_logits = mask_logits.sigmoid()

        if cfg is None:
            return poly_rbboxes, scores
        else:
            c_device = poly_rbboxes.device

            det_bboxes, det_labels, det_masks = multiclass_nms_polybbox_mask(poly_rbboxes,
                                                                                scores, mask_logits,
                                                                                cfg.score_thr, cfg.nms,
                                                                                cfg.max_per_img, img_h, img_w)

            return det_bboxes, det_labels, det_masks





