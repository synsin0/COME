import torch
import numpy as np
from tabulate import tabulate

from mmengine.dist import all_gather
from .base_evaluator import BaseEvaluator
from ..registry import EVALUATORS


@EVALUATORS.register_module()
class PQ(BaseEvaluator):

    def __init__(self, 
                 num_classes,
                 timestamps,
                 gt_sem_key='source_moving_sem',
                 gt_inst_key='source_moving_inst',
                 pred_sem_key='moving_sem_preds',
                 pred_inst_key='moving_inst_preds',
                 mode='2d',
                 index_offset=100000,
                 min_points=0,
                 matched_score=0.5,
                 dataset=None):
        super(PQ, self).__init__(dataset)
        self._name = 'PQ'
        self.num_classes = num_classes
        self.timestamps = timestamps
        self.gt_sem_key = gt_sem_key
        self.gt_inst_key = gt_inst_key
        self.pred_sem_key = pred_sem_key
        self.pred_inst_key = pred_inst_key
        self.mode = mode
        ## large number to seperate gt and pred index
        self.index_offset = index_offset
        ## minimum iou for a match to be valid tp, 0.5 for now
        self.matched_score = matched_score
        ## remove small instance?
        self.min_points = min_points
        ## reset all statistics
        ## default values
        self.eps = 1e-5

        self.clean()

    def clean(self):
        self.tp = torch.zeros((0, self.num_classes)).float().cuda()
        self.fp = torch.zeros((0, self.num_classes)).float().cuda()
        self.fn = torch.zeros((0, self.num_classes)).float().cuda()
        self.ious = torch.zeros((0, self.num_classes)).float().cuda()
        
        self.states = {}
    
    def prepare_data(self, inputs_dict, outputs_dict):
        gt_inst, gt_sem = inputs_dict[self.gt_inst_key], inputs_dict[self.gt_sem_key]
        if isinstance(gt_inst, np.ndarray):
            gt_inst = self.ious.new_tensor(gt_inst)
        if isinstance(gt_sem, np.ndarray):
            gt_sem = self.ious.new_tensor(gt_sem)
        assert isinstance(gt_inst, torch.Tensor) and isinstance(gt_sem, torch.Tensor)

        if self.mode == '2d':
            gt_inst, index = gt_inst.max(dim=-3)
            gt_sem = torch.gather(gt_sem, -3, index.unsqueeze(-3)).squeeze(-3)

        pred_inst = outputs_dict[self.pred_inst_key]
        pred_sem = outputs_dict[self.pred_sem_key]
        if isinstance(pred_inst, np.ndarray):
            pred_inst = self.ious.new_tensor(pred_inst)
        if isinstance(pred_sem, np.ndarray):
            pred_sem = self.ious.new_tensor(pred_sem)
        assert isinstance(pred_inst, torch.Tensor) and isinstance(pred_sem, torch.Tensor)

        return gt_inst, gt_sem, pred_inst, pred_sem

    def update(self, inputs_dict, outputs_dict):
        gt_inst, gt_sem, pred_inst, pred_sem = self.prepare_data(inputs_dict, outputs_dict)

        B, T = gt_inst.shape[:2]
        gt_inst = gt_inst[:, 0].reshape(B, -1).cpu().numpy()
        gt_sem = gt_sem[:, 0].reshape(B, -1).cpu().numpy()
        pred_inst = pred_inst[:, 0].reshape(B, -1).cpu().numpy()
        pred_sem = pred_sem[:, 0].reshape(B, -1).cpu().numpy()

        for i in range(B):
            tp_count, fp_count, fn_count, ious = self.single_batch_update(
                gt_inst[i]+1, gt_sem[i], pred_inst[i]+1, pred_sem[i])
            
            self.tp = torch.cat([self.tp, tp_count[None, ...]], dim=0)
            self.fp = torch.cat([self.fp, fp_count[None, ...]], dim=0)
            self.fn = torch.cat([self.fn, fn_count[None, ...]], dim=0)
            self.ious = torch.cat([self.ious, ious[None, ...]], dim=0)
    
    def single_batch_update(self, gt_inst, gt_sem, pred_inst, pred_sem):
        tp_count = torch.zeros((self.num_classes, )).float().cuda()
        fp_count = torch.zeros((self.num_classes, )).float().cuda()
        fn_count = torch.zeros((self.num_classes, )).float().cuda()
        cls_ious = torch.zeros((self.num_classes, )).float().cuda()

        for cls_id in range(self.num_classes):
           ### class mask
            cls_pred_inst_mask = (pred_sem == cls_id)
            cls_gt_inst_mask = (gt_sem == cls_id)

            ### instance data, make non-current-class 0
            cls_pred_inst = cls_pred_inst_mask * pred_inst
            cls_gt_inst = cls_gt_inst_mask * gt_inst

            ### generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(
                cls_pred_inst[cls_pred_inst > 0], return_counts=True
            )
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])

            ### generate the areas for each unique instance ground truth
            unique_gt, counts_gt = np.unique(
                cls_gt_inst[cls_gt_inst > 0], return_counts=True
            )
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])

            ### combination
            valid_comb_mask = np.logical_and(cls_pred_inst > 0, cls_gt_inst > 0)
            # traicky part... seperate different combinationt through a large number
            offset_comb = cls_pred_inst[valid_comb_mask] +\
                                    self.index_offset * cls_gt_inst[valid_comb_mask]
            unique_comb, counts_comb = np.unique(offset_comb, return_counts=True)

            ### generate intersection map,  > 0.5 considered as TP
            gt_labels = unique_comb // self.index_offset
            pred_labels = unique_comb % self.index_offset

            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])

            intersections = counts_comb
            unions = gt_areas + pred_areas - intersections

            ious = intersections.astype(float) / unions.astype(float)

            ### hard
            tp_indexes = ious > self.matched_score
            tp_count[cls_id] += np.sum(tp_indexes)
            cls_ious[cls_id] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

            ### count valid FN
            fn_count[cls_id] += np.sum(
                np.logical_and(
                    counts_gt >= self.min_points,
                    matched_gt == False
                )
            )

            ### count valid FP ## TODO? counts_pred >= self.min_points,
            fp_count[cls_id] += np.sum(
                np.logical_and(
                    counts_pred >= self.min_points,
                    matched_pred == False
                )
            )
        return tp_count, fp_count, fn_count, cls_ious

    def eval(self):
        tp = torch.stack(all_gather(self.tp), dim=1).flatten(end_dim=1)
        fp = torch.stack(all_gather(self.fp), dim=1).flatten(end_dim=1)
        fn = torch.stack(all_gather(self.fn), dim=1).flatten(end_dim=1)
        ious = torch.stack(all_gather(self.ious), dim=1).flatten(end_dim=1)

        self.tp = tp[:self._len]
        self.fp = fp[:self._len]
        self.fn = fn[:self._len]
        self.ious = ious[:self._len]

        tp = self.tp.sum(dim=0)
        fp = self.fp.sum(dim=0)
        fn = self.fn.sum(dim=0)
        ious = self.ious.sum(dim=0)

        sq_all = ious / tp.clip(min=self.eps)
        rq_all = tp / (tp + 0.5 * fn + 0.5 * fp).clip(min=self.eps)
        pq_all = sq_all * rq_all

        sq_mean = sq_all.mean(dim=-1)
        rq_mean = rq_all.mean(dim=-1)
        pq_mean = pq_all.mean(dim=-1)

        self.states[f'SQ'] = sq_mean.item()
        self.states[f'RQ'] = rq_mean.item()
        self.states[f'PQ'] = pq_mean.item()
        
        return self.states
    
    def format_string(self):
        headers = ['SQ', 'RQ', 'PQ']
        contents = []
        for prefix in ['SQ', 'RQ', 'PQ']:
            value = self.states[prefix]
            contents.append(f'{value*100:.2f}')
        return tabulate([contents], headers=headers, tablefmt='orgtbl')
