import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class LevelsetLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(LevelsetLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, mask_logits, targets, pixel_num):
        region_levelset_term = region_levelset()
        length_regu_term = length_regularization()

        region_levelset_loss = region_levelset_term(mask_logits, targets) / pixel_num
        length_regu = 0.00001 * length_regu_term(mask_logits) / pixel_num

        #loss_levelst = self.loss_weight * region_levelset_loss
        loss_levelst = self.loss_weight * region_levelset_loss + length_regu

        return loss_levelst


class region_levelset(nn.Module):
    '''
    The mian of region leveset function.
    '''

    def __init__(self):
        super(region_levelset, self).__init__()

        
    def forward(self, mask_score, lst_target):
        '''
        mask_score: predcited mask scores        tensor:(N,2,W,H) 
        lst_target:  input target for levelset   tensor:(N,C,W,H) 
        '''
        
        mask_score_f = mask_score[:, 0, :, :].unsqueeze(1)
        mask_score_b = mask_score[:, 1, :, :].unsqueeze(1)
        interior_ = torch.sum(mask_score_f * lst_target, (2, 3)) / torch.sum(mask_score_f, (2, 3)).clamp(min=0.00001)
        exterior_ = torch.sum(mask_score_b * lst_target, (2, 3)) / torch.sum(mask_score_b, (2, 3)).clamp(min=0.00001)
        interior_region_level = torch.pow(lst_target - interior_.unsqueeze(-1).unsqueeze(-1), 2)
        exterior_region_level = torch.pow(lst_target - exterior_.unsqueeze(-1).unsqueeze(-1), 2)
        region_level_loss = interior_region_level*mask_score_f + exterior_region_level*mask_score_b
        level_set_loss = torch.sum(region_level_loss, (1, 2, 3))/lst_target.shape[1]

        return level_set_loss


class length_regularization(nn.Module):

    '''
    calcaulate the length by the gradient for regularization.
    '''

    def __init__(self):
        super(length_regularization, self).__init__()

    def forward(self, mask_score):
        gradient_H = torch.abs(mask_score[:, :, 1:, :] - mask_score[:, :, :-1, :])
        gradient_W = torch.abs(mask_score[:, :, :, 1:] - mask_score[:, :, :, :-1])
        curve_length = torch.sum(gradient_H, dim=(1,2,3)) + torch.sum(gradient_W, dim=(1,2,3))
        return curve_length

