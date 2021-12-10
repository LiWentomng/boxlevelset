import torch
import torch.nn as nn

from ..registry import LOSSES

class region_levelset(nn.Module):
    '''
    the mian of leveset function
    '''

    def __init__(self):
        super(region_levelset, self).__init__()

    def forward(self, mask_score, norm_img, class_weight):
        '''
        mask_score: predcited mask scores tensor:(N,C,W,H)
        norm_img: normalizated images tensor:(N,C,W,H)
        class_weight: weight for different classes
        '''

        mask_score_shape = mask_score.shape
        norm_img_shape = norm_img.shape
        level_set_loss = 0.0

        for i in range(norm_img_shape[1]):

            norm_img_ = torch.unsqueeze(norm_img[:, i], 1)
            norm_img_ = norm_img_.expand(norm_img_shape[0], mask_score_shape[1], norm_img_shape[2], norm_img_shape[3])

            ave_similarity = torch.sum(norm_img_ * mask_score, (2, 3)) / torch.sum(mask_score, (2, 3))
            ave_similarity = ave_similarity.view(norm_img_shape[0], mask_score_shape[1], 1, 1)

            region_level = norm_img_ - ave_similarity.expand(norm_img_shape[0], mask_score_shape[1], norm_img_shape[2], norm_img_shape[3])
            region_level_loss = class_weight * region_level * region_level * mask_score
            level_set_loss += torch.sum(region_level_loss)

        return level_set_loss

class length_evolution(nn.Module):

    '''
    calcaulate the length of evolution curve by the gradient
    '''

    def __init__(self, func='l1'):
        super(length_evolution, self).__init__()
        self.func = func

    def forward(self, mask_score, class_weight):
        gradient_H = torch.abs(mask_score[:, :, 1:, :] - mask_score[:, :, :-1, :])
        gradient_W = torch.abs(mask_score[:, :, :, 1:] - mask_score[:, :, :, :-1])

        if (self.func == "l2"):
            gradient_H = gradient_H * gradient_H
            gradient_W = gradient_W * gradient_W

        curve_length = torch.sum(class_weight * gradient_H) + torch.sum(class_weight * gradient_W)
        return curve_length


class evolution_area(nn.Module):
    '''
    calcaulate the area of evolution curve
    '''
    def __init__(self):
        super(evolution_area, self).__init__()

    def forward(self, mask_score, class_weight):
        curve_area = torch.sum(class_weight * mask_score)
        return curve_area


@LOSSES.register_module
class LevelsetLoss(nn.Module):
    def __init__(self, levelset_evo_weight=0.000001, length_weight=0.00000001, area_weight=1.0):
        super(LevelsetLoss, self).__init__()
        self.levelset_evo_weight = levelset_evo_weight
        self.length_weight = length_weight
        self.area_weight = area_weight

    def forward(self, mask_logits, targets, class_weight):

        region_levelset_term = region_levelset()
        length_evolution_term = length_evolution()

        region_levelset_loss = region_levelset_term(mask_logits, targets, class_weight)
        length_regu = length_evolution_term(mask_logits, class_weight)

        loss_levelst =self.levelset_evo_weight * region_levelset_loss + self.length_weight * length_regu

        return loss_levelst


