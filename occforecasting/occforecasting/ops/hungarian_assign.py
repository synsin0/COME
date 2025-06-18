# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


class ClassificationCost:

    def __init__(self, weight=1):
        self.weight = weight

    def __call__(self, pred_scores, gt_labels):
        # cls_scores (..., num_queries, num_classes)
        # gt_labels (..., num_gts)
        gt_labels = gt_labels[..., None, :].expand(*pred_scores.shape[:-1], -1)
        pred_scores = pred_scores.softmax(-1)
        cls_cost = -torch.gather(pred_scores, -1, gt_labels)
        return cls_cost * self.weight


class DiceCost:

    def __init__(self, pred_act=False, eps=1e-3, naive_dice=True, weight=1):
        self.weight = weight
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice

    def _binary_mask_dice_loss(self, mask_preds, gt_masks):
        numerator = 2 * torch.matmul(mask_preds, gt_masks.transpose(-1, -2))
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[..., :, None] + \
                gt_masks.sum(-1)[..., None, :]
        else:
            denominator = mask_preds.pow(2).sum(1)[..., :, None] + \
                gt_masks.pow(2).sum(1)[..., None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, pred_masks, gt_masks):
        # pred_masks (..., num_queries, *)
        # gt_masks (..., num_gt, *)
        if self.pred_act:
            pred_masks = pred_masks.sigmoid()
        gt_masks = gt_masks.float()
        dice_cost = self._binary_mask_dice_loss(pred_masks, gt_masks)
        return dice_cost * self.weight


class CrossEntropyLossCost:

    def __init__(self,
                 use_sigmoid=True,
                 weight=1):
        self.weight = weight
        self.use_sigmoid = use_sigmoid

    def __call__(self, pred_masks, gt_masks):
        # pred_masks (..., num_queries, *)
        # gt_masks (..., num_gt, *)

        if not self.use_sigmoid:
            raise NotImplementedError

        gt_masks = gt_masks.float()
        n = pred_masks.shape[-1]
        pos = F.binary_cross_entropy_with_logits(
            pred_masks, torch.ones_like(pred_masks), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            pred_masks, torch.zeros_like(pred_masks), reduction='none')
        cls_cost = torch.matmul(pos, gt_masks.transpose(-1, -2)) + \
            torch.matmul(neg, (1 - gt_masks).transpose(-1, -2))
        cls_cost = cls_cost / n

        return cls_cost * self.weight
